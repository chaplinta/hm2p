from pathlib import Path
from paths.config import M2PConfig
from utils import misc as m2putils

class ProcPath:

    def __init__(self, cfg: M2PConfig, exp_id: str):

        self.cfg = cfg
        self.exp_id = exp_id

        self.raw_data_path = m2putils.get_exp_path(self.cfg.raw_path, exp_id)
        self.proc_data_path: Path = cfg.proc_path / exp_id
        self.proc_data_path.mkdir(exist_ok=True)

        self.proc_s2p_path: Path = cfg.s2p_path / self.exp_id
        self.proc_s2p_path.mkdir(exist_ok=True)

        self.video_path = self.cfg.video_path / self.exp_id
        self.video_path.mkdir(exist_ok=True)
        self.video_tracking_path = self.cfg.video_path / "tracking"
        self.video_tracking_path.mkdir(exist_ok=True)

        self.bad_2p_plots = self.cfg.bad_2p_plots / self.exp_id
        self.bad_2p_plots.mkdir(exist_ok=True)


        self.behave_file = self.proc_data_path / "behave.h5"
        self.behave_frames_file = self.proc_data_path / "behave_frames.h5"
        self.behave_events_file = self.proc_data_path / "behave_events.h5"

        self.pairs_file = self.proc_data_path / "somadend_pairs.h5"
        self.events_file = self.proc_data_path / "somadend_events.h5"





    # def get_behave_file(self, exp, cam_name="overhead"):
    #
    #     cam = exp.tracking_video.cameras[cam_name]
    #
    #     mov_fullfps_name = cam.file_name_base + "-cropped"
    #
    #     behav_file = self.behave_file / (mov_fullfps_name + ".filtered.metrics.h5")
    #
    #     return behav_file