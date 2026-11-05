file(REMOVE_RECURSE
  "CMakeFiles/ncnn-generate-llm-param"
  "param/hunyuan_0_5b_instruct_decoder_ncnn.hex.h"
  "param/hunyuan_0_5b_instruct_proj_out_ncnn.hex.h"
  "param/llama3_2_1b_decoder_ncnn.hex.h"
  "param/llama3_2_1b_proj_out_ncnn.hex.h"
  "param/minicpm4_0_5b_decoder_ncnn.hex.h"
  "param/minicpm4_0_5b_proj_out_ncnn.hex.h"
  "param/qwen2_5_0_5b_decoder_ncnn.hex.h"
  "param/qwen2_5_0_5b_proj_out_ncnn.hex.h"
  "param/qwen3_0_6b_decoder_ncnn.hex.h"
  "param/qwen3_0_6b_proj_out_ncnn.hex.h"
  "param/tinyllama_1_1b_decoder_ncnn.hex.h"
  "param/tinyllama_1_1b_proj_out_ncnn.hex.h"
  "param/youtu_llm_2b_decoder_ncnn.hex.h"
  "param/youtu_llm_2b_proj_out_ncnn.hex.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/ncnn-generate-llm-param.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
