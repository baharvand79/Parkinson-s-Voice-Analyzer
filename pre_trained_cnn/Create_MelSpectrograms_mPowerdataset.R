###############################################################################
# R code to generate Mel-scale spectrogram images from the mPower audio files #
###############################################################################

# Copyright (C) 2024 University of Arkansas for Medical Sciences
# Author: Yasir Rahmatallah, yrahmatallah@uams.edu
# Licensed under the Apache License, Version 2.0
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Code was tested using R version 4.1.2, and the following package versions:
# av_0.8.3, tuneR_1.4.0, oce_1.7-8, signal_0.7-7, viridis_0.6.2, torchaudio_0.3.1

# === 1. LIBRARIES ===
library(av)
library(tuneR)
library(oce)
library(signal)
library(viridis)
library(torchaudio) # Library for Mel-spectrograms

# === 2. CONFIGURATION ===
# -----------------------------------------------------------------------------
# The main input folder containing subfolders like "HC" and "PwPD"
base_input_dir  <- "D:/main_stream/voice_ut/just_start/data/mPower/copied_audio_files"

# The main output folder where all results will be saved
base_output_dir <- "D:/Projects/Voice/Parkinson-s-Voice-Analyzer/pre_trained_cnn/spectrogram_data"

# A name for the dataset, used for creating output folders
dataset_name <- "mPower"

# The names of the input group subfolders. MUST match the folder names exactly.
groups <- c("HC", "PwPD")
# -----------------------------------------------------------------------------

# === 3. HELPER FUNCTION (Unchanged) ===
trim_ends <- function(x, w=512, thr=0.5){
  len <- length(x)
  nseg <- floor(len/w)
  ste <- numeric(nseg)
  for(q in 1:nseg) ste[q] <- sum(x[((q-1)*w+1):(q*w)]^2)
  env <- rep(ste, each=w)
  check <- (env>thr)+0
  rr <- rle(check)
  if(length(rr$lengths)>1) {
    indmax <- which.max(rr$lengths * rr$values)
    if(indmax>1) {
      st <- sum(rr$lengths[1:(indmax-1)]) + 1
      en <- sum(rr$lengths[1:indmax])
    }
    if(indmax==1) {
      st <- 1
      en <- rr$lengths[1]
    }
  } else {st <- 1; en <- len}
  return(c(st, en))
}

# === 4. MAIN PROCESSING FUNCTION (REVISED FOR MEL-SPECTROGRAMS) ===
# This function saves files to the new directory structure
process_audio_group <- function(group_name, dataset_name, base_input_path, base_output_path) {
  
  message(paste("\n--- Processing Group:", group_name, "for Mel-Spectrograms ---"))
  
  # Map input group names (HC, PwPD) to desired output names (healthy, parkinson)
  output_group_name <- ifelse(group_name == "HC", "healthy", "parkinson")
  
  # Define final and intermediate output paths
  spec_output_path <- file.path(base_output_path, paste0(dataset_name, "_mel"), output_group_name)
  intermediate_path <- file.path(base_output_path, paste0(dataset_name, "_intermediate_files"))
  wav_output_path  <- file.path(intermediate_path, "wav", output_group_name)
  log_file_path    <- file.path(intermediate_path, paste0(group_name, "_mel_processing_log.csv"))
  
  # Create directories if they don't exist
  dir.create(wav_output_path, recursive = TRUE, showWarnings = FALSE)
  dir.create(spec_output_path, recursive = TRUE, showWarnings = FALSE)
  
  message(paste("Input folder:", base_input_path))
  message(paste("Saving Mel-Spectrograms to:", spec_output_path))
  
  # Find all .m4a files in the group's input directory
  fL <- list.files(path = base_input_path, pattern = ".m4a$", full.names = FALSE)
  if (length(fL) == 0) {
    message("No .m4a files found. Skipping.")
    return()
  }
  
  # Initialize logging and progress tracking
  file_names <- gsub(fL, pattern = ".m4a", replacement = "")
  total_files <- length(file_names)
  processed_count <- 0
  log_data <- data.frame(
    filename = character(),
    status = character(),
    reason = character(),
    stringsAsFactors = FALSE
  )
  
  message(paste("Found", total_files, "files to process."))
  
  # --- Start the loop for this group ---
  for (fn in file_names) {
    processed_count <- processed_count + 1
    file_m4a <- file.path(base_input_path, paste0(fn, ".m4a"))
    file_wav <- file.path(wav_output_path, paste0(fn, ".wav"))
    file_jpg <- file.path(spec_output_path, paste0(fn, ".jpg"))
    
    tryCatch({
      if (!file.exists(file_wav)) {
        av_audio_convert(file_m4a, file_wav, channels = 1, verbose = FALSE)
      }
      time.limit <- 1.5
      ds <- 5
      train_audio <- readWave(file_wav)
      audio.nor <- tuneR::normalize(train_audio, unit = "32", pcm = FALSE)
      x <- audio.nor@left
      fs <- train_audio@samp.rate
      fs <- fs / ds
      x <- x[seq(1, length(x), by = ds)]
      if (length(x) < 26501) {
        stop("File is too short for the initial hard-coded trim.")
      }
      x <- x[16500:(length(x) - 10000)]
      ends <- trim_ends(x, w = 512, thr = 0.5)
      mp <- floor(mean(ends))
      
      if ((ends[2] - ends[1]) < (fs * time.limit)) {
        log_data <- rbind(log_data, data.frame(filename = fn, status = "Skipped", reason = "Duration < 1.5s"))
      } else {
        x <- x[(mp - (fs * time.limit / 2)):(mp + (fs * time.limit / 2))]
        
        # --- Mel-Spectrogram Generation using torchaudio ---
        audio.nor@left <- x
        audio.nor@samp.rate <- fs
        
        sample_tensor <- transform_to_tensor(audio.nor)
        
        mel_spec_transform <- transform_mel_spectrogram(
          sample_rate = sample_tensor[[2]],
          n_fft = 1024, win_length = 512, hop_length = 51, # ~90% overlap
          f_min = 0, f_max = 4000, n_mels = 256,
          window_fn = torch::torch_hann_window,
          power = 2, normalized = FALSE
        )
        
        mel_spec_tensor <- mel_spec_transform(sample_tensor[[1]])
        spec_array <- as.array(mel_spec_tensor$log2()[1]$t())
        
        # --- Plotting and Saving ---
        jpeg(filename = file_jpg, res = 300, width = 2, height = 2, units = "in", pointsize = 1, quality = 100)
        par(mfrow = c(1, 1), mar = c(0, 0, 0, 0))
        image(spec_array, col = oce.colorsViridis(256), xaxt = "n", yaxt = "n")
        dev.off()
        
        log_data <- rbind(log_data, data.frame(filename = fn, status = "Success", reason = "Mel-spectrogram created"))
      }
    }, error = function(e) {
      log_data <<- rbind(log_data, data.frame(filename = fn, status = "Error", reason = conditionMessage(e)))
    })
    cat(sprintf("\rProcessing: %d / %d (%s)", processed_count, total_files, group_name))
  }
  
  # Finalize and save the log file
  cat("\n")
  write.csv(log_data, file = log_file_path, row.names = FALSE)
  message(paste("--- Finished processing", group_name, "---"))
  message(paste("Log file saved to:", log_file_path))
}

# === 5. EXECUTION LOOP ===
# This loop runs the main function for each group defined in the configuration
for (group in groups) {
  current_input_path <- file.path(base_input_dir, group)
  
  process_audio_group(
    group_name = group,
    dataset_name = dataset_name,
    base_input_path = current_input_path,
    base_output_path = base_output_dir
  )
}

message("\nAll processing complete.")