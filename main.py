#main.py
import sys
import json
import os
from video_extractor import extract_video_audio
from resource_helper import get_resource_path, log_environment_info

def report_progress(message):
    print(message, flush=True)

def report_error(message):
    print(json.dumps({"type": "error", "message": message}), flush=True)
    sys.exit(1)

def report_result(data):
    print(json.dumps({"type": "result", "data": data}), flush=True)

def get_output_directory():
    downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    output_dir = os.path.join(downloads_dir, 'VCFS_Output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_model_path():
    # Use the resource helper to find the model file
    return get_resource_path('ResNet/best_model234.pth')

def main():
    try:
        # Log environment info for debugging
        log_environment_info()
        
        if len(sys.argv) < 2:
            report_error("Please provide the input file path as an argument")
        
        input_file = sys.argv[1]
        output_dir = get_output_directory()
        # Use resource helper for profanity list path
        profanity_file_path = get_resource_path("profanity_list.txt")
        model_path = get_model_path()
        
        report_progress(f"Starting video processing with:")
        report_progress(f"Input: {input_file}")
        report_progress(f"Output: {output_dir}")
        report_progress(f"Profanity file: {profanity_file_path}")
        report_progress(f"Model: {model_path}")
        
        # Process video using video_extractor only
        result = extract_video_audio(
            input_file, 
            output_dir, 
            report_progress,
            profanity_file_path
        )

        if isinstance(result, dict) and 'error' in result:
            report_error(result['error'])
        else:
            report_result(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        report_error(f"An unexpected error occurred: {str(e)}\n{error_details}")

if __name__ == "__main__":
    main()