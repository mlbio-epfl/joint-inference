import os

TEXT_INPUT = "text_input"
TEXT_PHI = "text_phi"
PIXELS_INPUT = "pixels_input"
PIXELS_PHI = "pixels_phi"
PIXELS_PHI_SEQ = "pixels_phi_seq"
INPUT_ID = "input_id"

# for consitency with huggingface default cache dir
CACHE_DIR = os.path.join(os.environ['XDG_CACHE_HOME'], 'huggingface', 'hub')
LOC_FINDER_TOKEN = "<loc_finder>"