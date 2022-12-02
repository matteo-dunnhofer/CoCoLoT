import vot_path
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = vot_path.base_path+'/mlpLT/Stark/data/got10k_lmdb'
    settings.got10k_path = vot_path.base_path+'/mlpLT/Stark/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = vot_path.base_path+'/mlpLT/Stark/data/lasot_lmdb'
    settings.lasot_path = vot_path.base_path+'/mlpLT/Stark/data/lasot'
    settings.network_path = vot_path.base_path+'/mlpLT/Stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = vot_path.base_path+'/mlpLT/Stark'
    settings.result_plot_path = vot_path.base_path+'/mlpLT/Stark/test/result_plots'
    settings.results_path = vot_path.base_path+'/mlpLT/Stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = vot_path.base_path+'/mlpLT/Stark'
    settings.segmentation_path = vot_path.base_path+'/mlpLT/Stark/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = vot_path.base_path+'/mlpLT/Stark/data/trackingNet'
    settings.uav_path = ''
    settings.vot_path = vot_path.base_path+'/mlpLT/Stark/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

