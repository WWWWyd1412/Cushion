import serial
import struct
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter

# Configurations
PORT = "COM10"          # Replace with your actual COM port (e.g. COM3 or /dev/ttyUSB0)
BAUD = 921600          # Matches the new baud rate in firmware
HEADER = b'\xAA\xBB'
FOOTER = b'\xCC\xDD'
FRAME_SIZE = 2048       # 32x32 * 2 bytes (uint16_t raw ADC values)

def adc_to_pressure(adc_val):
    # F = (128 * ad_temp - 75000) / 96000 - 0.26
    # F = round(F * 100) / 100
    f = (128.0 * adc_val - 75000.0) / 96000.0 - 0.26
    return round(f, 2)

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0)
    except Exception as e:
        print(f"Error opening serial port {PORT}: {e}")
        return
        
    ser.reset_input_buffer()
    
    # Open raw data log file
    log_filename = "raw_adc_data.txt"
    try:
        raw_data_file = open(log_filename, "w", encoding="utf-8")
        print(f"Logging raw ADC values to {log_filename}")
    except Exception as e:
        print(f"Error opening log file: {e}")
        ser.close()
        return

    # Send Command 4 to start binary streaming
    print(f"Successfully opened {PORT} at {BAUD} bps.")
    print("Sending trigger command 4 (Start Binary Streaming)...")
    ser.write(b'\x04')
    
    frame_count = 0
    start_time = time.time()
    
    # Set up real-time matplotlib heatmap
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    matrix_data = np.zeros((32, 32))
    im = ax.imshow(matrix_data, cmap='jet', interpolation='nearest', vmin=0, vmax=2)
    fig.colorbar(im, ax=ax, label='Pressure (N)')
    ax.set_title("Real-Time 32x32 Pressure Heatmap")
    
    # Track whether the matplotlib window is open
    is_open = [True]
    def on_close(event):
        is_open[0] = False
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.show(block=False)
    
    try:
        while True:
            # Check if plot window was closed
            if not is_open[0]:
                print("Plot window closed. Exiting...")
                break

            # Find frame header
            header_candidate = ser.read(2)
            if header_candidate != HEADER:
                continue
                
            # Read payload + footer
            payload = ser.read(FRAME_SIZE)
            footer = ser.read(2)
            
            if footer != FOOTER:
                print("Frame sync lost (invalid footer)")
                continue
                
            # Unpack 1024 uint16 values (little endian)
            raw_values = struct.unpack('<1024H', payload)
            
            # Convert to pressure values
            pressure_matrix = [adc_to_pressure(val) for val in raw_values]
            matrix_2d = np.array(pressure_matrix).reshape((32, 32))
            
            frame_count += 1
            
            # Save raw ADC data to file: time_offset followed by 1024 raw values
            t_offset = time.time() - start_time
            raw_data_file.write(f"{t_offset:.4f} {' '.join(map(str, raw_values))}\n")
            if frame_count % 10 == 0:
                raw_data_file.flush()
            
            # Spatial Filtering (Median filter only to remove noise spikes, keeping grid sharp)
            filtered_matrix = median_filter(matrix_2d, size=3)
            
            # Update heatmap in UI, throttled to every 2 frames for performance
            if frame_count % 2 == 0 and is_open[0]:
                im.set_data(filtered_matrix)
                # Dynamically set limits based on actual range, ensuring a minimum span of 2.0
                min_val = np.min(filtered_matrix)
                max_val = np.max(filtered_matrix)
                span = max_val - min_val
                if span < 2.0:
                    max_val = min_val + 2.0
                im.set_clim(min_val, max_val)
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # Print frame rate stats every 10 frames
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"--- Frame {frame_count} ---")
                print(f"Current Frame Rate: {fps:.2f} Hz")
                # Print a 4x4 matrix from the center of the 32x32 cushion (using original raw pressure values)
                print("Center 4x4 Pressure values (N, Raw):")
                center_subgrid = matrix_2d[14:18, 14:18]
                for row in center_subgrid:
                    print("  " + "  ".join(f"{val:6.2f}" for val in row))
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting...")
    finally:
        # Close raw data file
        try:
            raw_data_file.close()
            print("Raw ADC data logged and file closed.")
        except Exception as e:
            print(f"Error closing log file: {e}")
            
        # Send Command 2 to stop
        print("\nStopping streaming...")
        try:
            ser.write(b'\x02')
        except Exception as e:
            print(f"Failed to send stop command: {e}")
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()


