################################################################################
# R code to generate linear-scale spectrogram images from the UAMS audio files #
################################################################################

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
# av_0.8.3, tuneR_1.4.0, oce_1.7-8, signal_0.7-7, viridis_0.6.2                    

# === 1. LIBRARIES ===
library(tuneR)
library(oce)
library(signal)
library(viridis)

# === 2. CONFIGURATION ===
# -----------------------------------------------------------------------------
# The main input folder containing subfolders like "HC" and "PwPD"
base_input_dir  <- "D:/main_stream/voice_ut/just_start/data/UAMS"

# The main output folder where all results will be saved
# base_output_dir <- "G:/My Drive/Scripts/pretrained_cnn/spectrogram_data"
base_output_dir <- "D:/Projects/Voice/Parkinson-s-Voice-Analyzer/pre_trained_cnn/spectrogram_data"

# A name for the dataset, used for creating output folders
dataset_name <- "UAMS"

# The names of the input group subfolders. MUST match the folder names exactly.
groups <- c("HC", "PwPD") # Or "healthy", "parkinson" if your folders are named that way
# -----------------------------------------------------------------------------

# === 3. HELPER FUNCTION ===
# Function to trim silence parts from the start and end of the recording
trim_ends <- function(x, w = 100, thr = 1) {
  len <- length(x)
  env <- numeric(length = len - w)
  for (k in 1:(len - w)) {
    env[k] <- sum(x[k:(k + w)]^2)
  }
  ind <- which(env > thr)
  if (length(ind) == 0) {
    # If no part of the signal is above the threshold, return the full length
    return(c(1, len))
  }
  st <- ind[1]
  en <- tail(ind, n = 1)
  return(c(st, en))
}

# === 4. MAIN PROCESSING FUNCTION  ===
# This function processes all files for a single group
process_audio_group <- function(group_name, dataset_name, base_input_path, base_output_path) {
  
  message(paste("\n--- Processing Group:", group_name, "for UAMS Linear-Spectrograms ---"))
  
  # Map input group names to desired output names
  output_group_name <- ifelse(group_name == "HC", "healthy", "parkinson")
  
  # Define final and intermediate output paths
  spec_output_path <- file.path(base_output_path, paste0(dataset_name, "_linear"), output_group_name)
  intermediate_path <- file.path(base_output_path, paste0(dataset_name, "_intermediate_files"))
  log_file_path    <- file.path(intermediate_path, paste0(group_name, "_linear_processing_log.csv"))
  
  # Create directories if they don't exist
  dir.create(spec_output_path, recursive = TRUE, showWarnings = FALSE)
  dir.create(intermediate_path, recursive = TRUE, showWarnings = FALSE) # For the log file
  
  message(paste("Input folder:", base_input_path))
  message(paste("Saving Spectrograms to:", spec_output_path))
  
  # Find all .wav files in the group's input directory
  fL <- list.files(path = base_input_path, pattern = ".wav$", full.names = FALSE)
  if (length(fL) == 0) {
    message("No .wav files found. Skipping.")
    return()
  }
  
  # Initialize logging and progress tracking
  file_names <- gsub(fL, pattern = ".wav", replacement = "")
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
    file_wav <- file.path(base_input_path, paste0(fn, ".wav"))
    file_jpg <- file.path(spec_output_path, paste0(fn, ".jpg"))
    
    tryCatch({
      time.limit <- 1.5
      
      train_audio <- readWave(file_wav)
      audio.nor <- tuneR::normalize(train_audio, unit = "32", pcm = FALSE)
      x <- audio.nor@left
      fs <- train_audio@samp.rate
      
      ends <- trim_ends(x, w = 100, thr = 1)
      
      # Check if the remaining duration is sufficient
      if ((ends[2] - ends[1]) < (fs * time.limit)) {
        log_data <- rbind(log_data, data.frame(filename = fn, status = "Skipped", reason = "Duration < 1.5s after trim"))
      } else {
        # Clip 1.5s from the start of the detected sound
        x <- x[ends[1]:(ends[1] + (fs * time.limit) - 1)]
        
        # --- Spectrogram Generation ---
        spec <- specgram(x = x, n = 1024, Fs = fs, window = hanning(1024), overlap = 768)
        P <- abs(spec$S)
        P <- P / max(P)
        P <- 10 * log10(P)
        t <- spec$t
        
        # --- Plotting and Saving ---
        jpeg(filename = file_jpg, res = 300, width = 2, height = 2, units = "in", pointsize = 1, quality = 100)
        imagep(x = t, y = spec$f, z = t(P), col = oce.colorsViridis(256), ylim = c(0, 4000),
               axes = FALSE, ylab = "", xlab = "", drawPalette = F, decimate = F, mar = c(0, 0, 0, 0))
        dev.off()
        
        log_data <- rbind(log_data, data.frame(filename = fn, status = "Success", reason = "Spectrogram created"))
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