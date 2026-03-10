#!/usr/bin/env python3
"""Lock down the hm2p security group to allow access only from a specific IP.

Adds inbound rules for SSH (22) and Streamlit (8501) from a single IP address
(IPv4 or IPv6), and removes any existing wide-open (0.0.0.0/0 or ::/0) rules
on port 8501.

Run this from your local machine (not the devcontainer — IAM is blocked there).

Usage:
    uv run scripts/setup_sg_lockdown.py --ip 103.106.88.142
    uv run scripts/setup_sg_lockdown.py --ip 2001:db8::1
    uv run scripts/setup_sg_lockdown.py --dry-run --ip 103.106.88.142
    uv run scripts/setup_sg_lockdown.py --teardown --ip 103.106.88.142
"""

from __future__ import annotations

import argparse
import ipaddress
import sys

DEFAULT_SG_ID = "sg-020161fb424325e6b"
DEFAULT_IP = "103.106.88.142"
REGION = "ap-southeast-2"

# Ports to add rules for
PORTS = [
    (22, "SSH"),
    (8501, "Streamlit"),
]


def _is_ipv6(ip: str) -> bool:
    """Check if the given IP is IPv6."""
    try:
        return isinstance(ipaddress.ip_address(ip), ipaddress.IPv6Address)
    except ValueError:
        return False


def _cidr(ip: str) -> str:
    """Return a CIDR for the IP (/32 for IPv4, /128 for IPv6)."""
    if _is_ipv6(ip):
        return f"{ip}/128"
    return f"{ip}/32"


def _build_permissions(ip: str) -> list[dict]:
    """Build the IpPermissions list for the target ports."""
    cidr = _cidr(ip)
    is_v6 = _is_ipv6(ip)
    perms = []
    for port, desc in PORTS:
        perm: dict = {
            "IpProtocol": "tcp",
            "FromPort": port,
            "ToPort": port,
        }
        if is_v6:
            perm["Ipv6Ranges"] = [{"CidrIpv6": cidr, "Description": f"{desc} from allowed IP"}]
        else:
            perm["IpRanges"] = [{"CidrIp": cidr, "Description": f"{desc} from allowed IP"}]
        perms.append(perm)
    return perms


def _find_wide_open_8501(sg: dict) -> list[dict]:
    """Find existing inbound rules on port 8501 that allow 0.0.0.0/0 or ::/0."""
    to_remove = []
    for perm in sg.get("IpPermissions", []):
        from_port = perm.get("FromPort", 0)
        to_port = perm.get("ToPort", 0)
        if not (from_port <= 8501 <= to_port):
            continue

        wide_v4 = [r for r in perm.get("IpRanges", []) if r.get("CidrIp") == "0.0.0.0/0"]
        wide_v6 = [r for r in perm.get("Ipv6Ranges", []) if r.get("CidrIpv6") == "::/0"]

        if wide_v4:
            to_remove.append({
                "IpProtocol": perm["IpProtocol"],
                "FromPort": from_port,
                "ToPort": to_port,
                "IpRanges": wide_v4,
            })
        if wide_v6:
            to_remove.append({
                "IpProtocol": perm["IpProtocol"],
                "FromPort": from_port,
                "ToPort": to_port,
                "Ipv6Ranges": wide_v6,
            })

    return to_remove


def dry_run(sg_id: str, ip: str) -> None:
    """Print what would change without making changes."""
    cidr = _cidr(ip)
    print(f"# Security group: {sg_id}")
    print(f"# Allowed IP:     {cidr}")
    print()
    print("# 1. Remove any wide-open rules on port 8501")
    print()
    for port, desc in PORTS:
        print(f"# 2. Add {desc} (port {port}) from {cidr}")
    print()


def create(sg_id: str, ip: str) -> None:
    """Lock down the security group."""
    import boto3

    ec2 = boto3.client("ec2", region_name=REGION)
    cidr = _cidr(ip)

    try:
        resp = ec2.describe_security_groups(GroupIds=[sg_id])
        sg = resp["SecurityGroups"][0]
        print(f"Found security group: {sg.get('GroupName', sg_id)}")
    except Exception as e:
        print(f"ERROR: Could not describe security group {sg_id}: {e}")
        sys.exit(1)

    # Step 1: Remove wide-open rules on port 8501
    wide_open = _find_wide_open_8501(sg)
    if wide_open:
        for rule in wide_open:
            try:
                ec2.revoke_security_group_ingress(
                    GroupId=sg_id, IpPermissions=[rule]
                )
                src = (rule.get("IpRanges") or rule.get("Ipv6Ranges", []))[0]
                src_cidr = src.get("CidrIp", src.get("CidrIpv6", "unknown"))
                print(f"  Removed wide-open rule: port 8501 from {src_cidr}")
            except Exception as e:
                print(f"  Warning: could not remove rule: {e}")
    else:
        print("  No wide-open rules on port 8501 found")

    # Step 2: Add restricted rules
    permissions = _build_permissions(ip)
    for perm in permissions:
        port = perm["FromPort"]
        desc = next(d for p, d in PORTS if p == port)
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id, IpPermissions=[perm]
            )
            print(f"  Added rule: port {port} ({desc}) from {cidr}")
        except Exception as e:
            if "InvalidPermission.Duplicate" in str(e):
                print(f"  Rule already exists: port {port} ({desc}) from {cidr}")
            else:
                print(f"  ERROR adding port {port} rule: {e}")

    print(f"\nDone! Security group {sg_id} now allows ports 22 and 8501 from {cidr}.")


def teardown(sg_id: str, ip: str) -> None:
    """Remove the rules added by this script."""
    import boto3

    ec2 = boto3.client("ec2", region_name=REGION)
    permissions = _build_permissions(ip)
    cidr = _cidr(ip)

    for perm in permissions:
        port = perm["FromPort"]
        desc = next(d for p, d in PORTS if p == port)
        try:
            ec2.revoke_security_group_ingress(
                GroupId=sg_id, IpPermissions=[perm]
            )
            print(f"Removed rule: port {port} ({desc}) from {cidr}")
        except Exception as e:
            if "InvalidPermission.NotFound" in str(e):
                print(f"Rule not found (already removed): port {port} ({desc}) from {cidr}")
            else:
                print(f"Warning: could not remove port {port} rule: {e}")

    print(f"\nTeardown complete for {sg_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lock down hm2p security group to a single IP address",
    )
    parser.add_argument(
        "--sg-id", default=DEFAULT_SG_ID,
        help=f"Security group ID (default: {DEFAULT_SG_ID})",
    )
    parser.add_argument(
        "--ip", default=DEFAULT_IP,
        help=f"IP address to allow, IPv4 or IPv6 (default: {DEFAULT_IP})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without making changes",
    )
    group.add_argument(
        "--teardown", action="store_true",
        help="Remove rules added by this script",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.sg_id, args.ip)
    elif args.teardown:
        teardown(args.sg_id, args.ip)
    else:
        create(args.sg_id, args.ip)


if __name__ == "__main__":
    main()
