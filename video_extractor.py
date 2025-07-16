#Video extractor.py
import os
import subprocess
from audio_processor import process_audio
from video_processor import process_video
import datetime
import uuid
from resource_helper import get_resource_path, log_environment_info

def generate_unique_filename(basename, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{basename}{timestamp}_{unique_id}{extension}"


def extract_video_audio(input_file, output_dir, progress_callback, profanity_file_path):
    input_filename = os.path.splitext(os.path.basename(input_file))[0]

    # Temporary files
    video_output_file = os.path.join(output_dir, generate_unique_filename(f"{input_filename}_extracted", ".mp4"))
    audio_output_file = os.path.join(output_dir, generate_unique_filename(f"{input_filename}_extracted", ".mp3"))
    filtered_audio_file = os.path.join(output_dir, generate_unique_filename(f"{input_filename}_filtered", ".mp3"))
    filtered_video_file = os.path.join(output_dir, generate_unique_filename(f"{input_filename}_filtered_video", ".mp4"))
    final_output_file = os.path.join(output_dir, generate_unique_filename(f"{input_filename}_final", ".mp4"))

    if not os.path.exists(input_file):
        return {"error": f"Input file does not exist: {input_file}"}

    if not input_file.lower().endswith('.mp4'):
        return {"error": "File is not in MP4 format"}

    try:
        # Extract audio - 20% of total progress
        progress_callback("Extracting audio from video (0%)")
        subprocess.run([
            'ffmpeg', '-i', input_file,
            '-vn', '-acodec', 'libmp3lame',
            '-q:a', '2', audio_output_file
        ], check=True)
        progress_callback("Audio extraction complete (20%)")

        # Extract video - 20% of total progress
        progress_callback("Extracting video stream (20%)")
        subprocess.run([
            'ffmpeg', '-i', input_file,
            '-an',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            video_output_file
        ], check=True)
        progress_callback("Video extraction complete (40%)")

        # Process audio - 20% of total progress
        def audio_progress(msg):
            if "Processing words" in msg:
                # Extract percentage from the message
                try:
                    percent = float(msg.split(":")[1].strip().replace("%", ""))
                    overall_percent = 40 + (percent * 0.2)  # Scale to 20% of total progress
                    progress_callback(f"Processing audio: {percent:.1f}% (Overall: {overall_percent:.1f}%)")
                except:
                    progress_callback(f"Processing audio: {msg}")
            else:
                progress_callback(f"Processing audio: {msg}")

        process_result = process_audio(audio_output_file, filtered_audio_file, audio_progress, profanity_file_path)
        if 'error' in process_result:
            return process_result
        progress_callback("Audio processing complete (60%)")

        # Process video - 20% of total progress
        def video_progress(msg):
            if "Analyzing video" in msg:
                try:
                    percent = float(msg.split(":")[1].strip().replace("%", ""))
                    overall_percent = 60 + (percent * 0.2)  # Scale to 20% of total progress
                    progress_callback(f"Processing video: {percent:.1f}% (Overall: {overall_percent:.1f}%)")
                except:
                    progress_callback(f"Processing video: {msg}")
            else:
                progress_callback(f"Processing video: {msg}")
        
        model_path = get_resource_path('ResNet/best_model234.pth')

        video_result = process_video(
            video_output_file,
            filtered_video_file,
            model_path,
            video_progress
        )

        if not video_result['success']:
            return {"error": f"Video processing failed: {video_result.get('error', 'Unknown error')}"}
        progress_callback("Video processing complete (80%)")

        # Final combination - 20% of total progress
        progress_callback("Combining processed audio and video (80%)")
        subprocess.run([
            'ffmpeg',
            '-i', filtered_video_file,
            '-i', filtered_audio_file,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-strict', 'experimental',
            '-map', '0:v:0',
            '-map', '1:a:0',
            final_output_file
        ], check=True)
        progress_callback("Final video creation complete (100%)")

        # Verify the output file
        probe_command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            final_output_file
        ]
        
        video_stream = subprocess.run(probe_command, capture_output=True, text=True)
        
        if not video_stream.stdout.strip():
            return {"error": "Final output file has no video stream"}

        # Clean up intermediate files
        for file in [video_output_file, audio_output_file, filtered_audio_file, filtered_video_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass

        return {
            "success": True,
            "video": final_output_file,
            "status": "success",
            "message": "Your video has been successfully processed and filtered."
        }

    except subprocess.CalledProcessError as e:
        return {"error": f"Error in FFmpeg command: {str(e)}"}
    except Exception as e:
        return {"error": f"Error in video extraction: {str(e)}"}
    finally:
        # Ensure cleanup of temporary files in case of error
        for file in [video_output_file, audio_output_file, filtered_audio_file, filtered_video_file]:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass