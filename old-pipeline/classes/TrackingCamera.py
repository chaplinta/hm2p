import os
class TrackingCamera:

    def __init__(self, name, device, daqchanname, daqphyschan, is_master, resolution, file):
        self.name = name
        self.device = device
        self.daqchanname = daqchanname
        self.daqphyschan = daqphyschan
        self.is_master = is_master
        self.resolution = resolution
        self.file = file
        (file_path, file_name) = os.path.split(file)
        self.file_path = file_path
        self.file_name = file_name
        (file_name_base, file_ext) = os.path.splitext(file_name)
        self.file_name_base = file_name_base
        self.file_ext = file_ext

        self.frm_event_indexes = None
        self.frm_event_times = None
        self.out_event_indexes = None
        self.out_event_times = None

        self.synced_events = []


    def add_daq_data(self, frm_events_indexes, frm_event_times, out_events_indexes, out_event_times):
        self.frm_event_indexes = frm_events_indexes
        self.frm_event_times = frm_event_times
        self.out_event_indexes = out_events_indexes
        self.out_event_times = out_event_times
