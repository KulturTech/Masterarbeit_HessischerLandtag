import time
import re

# Monitor the training output file
output_file = r'C:\Users\gsera\AppData\Local\Temp\claude\c--Users-gsera-OneDrive-Desktop-Masterarbeit-Masterarbeit-HessischerLandtag\tasks\bff6e51.output'

print("Starting training monitor...")
print("Will notify at: 25%, 50%, 75%, and 100%")

last_progress = 0
milestones = [25, 50, 75, 100]
notified = set()

while True:
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Look for progress pattern like "  20%|##        | 11/55"
        matches = re.findall(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)', content)

        if matches:
            # Get the most recent progress
            progress_pct, current, total = matches[-1]
            progress_pct = int(progress_pct)
            current = int(current)
            total = int(total)

            # Check if we've crossed any milestones
            for milestone in milestones:
                if progress_pct >= milestone and milestone not in notified:
                    print(f"\n{'='*60}")
                    print(f"MILESTONE REACHED: {milestone}%")
                    print(f"Progress: {current}/{total} steps completed")
                    print(f"{'='*60}\n")
                    notified.add(milestone)

            # Check if training is complete
            if 'TRAINING COMPLETE' in content or current == total:
                print(f"\n{'='*60}")
                print("TRAINING COMPLETE!")
                print(f"All {total} steps finished")
                print(f"{'='*60}\n")
                break

        # Wait before checking again
        time.sleep(30)  # Check every 30 seconds

    except FileNotFoundError:
        print("Training output file not found, waiting...")
        time.sleep(30)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
