# import the required libraries for lip reading model
import cv2, os, sys
import tempfile
from argparse import Namespace
# add the avhubert repository path to the system path --required for the following imports--
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "av_hubert/avhubert")
sys.path.append(repo_path)
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
import whisper # import speech-to-text model
from difflib import SequenceMatcher # import the function to calculate sentence similarity

if len(sys.argv) == 1:
    # Append a dummy argument to avoid model duplication error
    sys.argv.append("dummy_arg_to_prevent_error")

# function to run the inference code on the lip reading model
def predict(video_path, ckpt_path, user_dir):
  num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  data_dir = tempfile.mkdtemp()
  tsv_cont = ["\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
  label_cont = ["DUMMY\n"]
  with open(f"{data_dir}/test.tsv", "w") as fo:
    fo.write("".join(tsv_cont))
  with open(f"{data_dir}/test.wrd", "w") as fo:
    fo.write("".join(label_cont))
  utils.import_user_module(Namespace(user_dir=user_dir))
  modalities = ["video"]
  gen_subset = "test"
  gen_cfg = GenerationConfig(beam=20)
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  models = [model.eval().cpu() for model in models]
  saved_cfg.task.modalities = modalities
  saved_cfg.task.data = data_dir
  saved_cfg.task.label_dir = data_dir
  task = tasks.setup_task(saved_cfg.task)
  task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
  generator = task.build_generator(models, gen_cfg)

  def decode_fn(x):
      dictionary = task.target_dictionary
      symbols_ignore = generator.symbols_to_strip_from_output
      symbols_ignore.add(dictionary.pad())
      return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

  itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
  sample = next(itr)
  #sample = utils.move_to_cuda(sample)
  hypos = task.inference_step(generator, models, sample)
  ref = decode_fn(sample['target'][0].int().cpu())
  hypo = hypos[0][0]['tokens'].int().cpu()
  hypo = decode_fn(hypo)
  return hypo

# function to return the text output of the lip reading model
def get_text_from_lip_reading(roi_path):
    mouth_roi_path, ckpt_path = roi_path, "models/finetune-model.pt" # checkpoint of the finetune model
    user_dir = "av_hubert/avhubert" # the directory to the model
    hypo = predict(mouth_roi_path, ckpt_path, user_dir)
    return hypo

# function to get text output of the speech-to-text model
def get_text_from_stt(video_path):
   # load the model
   model = whisper.load_model("medium") 
   # transcribe the audio
   result = model.transcribe(video_path)
   # return the text result
   return result["text"]

"""
function to calculate the similarity between text from lip reading model
and text from speech-to-text model
"""
def classify_input(lip_reading_text, stt_text):
    similarity = SequenceMatcher(None, lip_reading_text, stt_text).ratio() * 100
    # classify the result as Real/Fake
    if similarity > 50: label = "Real"  
    else: 
      label = "Fake" 
      similarity = 100 - similarity
    return similarity, label
