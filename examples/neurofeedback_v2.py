import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import utils
import pandas as pd
from datetime import datetime

# Define Band class
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

# Experimental parameters
BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

# Initialize data structures
band_history = {'Time': [], 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [],
                'Alpha Relaxation': [], 'Beta Concentration': [], 'Theta Relaxation': []}
plt.ion()  # Interactive mode for real-time plotting

if __name__ == "__main__":
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    fs = int(inlet.info().nominal_srate())

    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))

    # Get the current date and time for filenames
    current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_filename = f"{current_time_str}_eeg_plot.png"
    csv_filename = f"{current_time_str}_eeg_data.csv"

    # Set up the plots
    fig, (ax1, ax2) = plt.subplots(2, 1)
    lines_bands = {band: ax1.plot([], [], label=band)[0] for band in ['Delta', 'Theta', 'Alpha', 'Beta']}
    lines_metrics = {metric: ax2.plot([], [], label=metric)[0] for metric in ['Alpha Relaxation', 'Beta Concentration', 'Theta Relaxation']}
    ax1.legend()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Band Power')
    ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Metric Value')

    print('Press Ctrl-C in the console to break the while loop.')
    try:
        while True:
            eeg_data, timestamps = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            if eeg_data:
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
                eeg_buffer, filter_state = utils.update_buffer(eeg_buffer, ch_data, notch=True, filter_state=filter_state)

                data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
                band_powers = utils.compute_band_powers(data_epoch, fs)
                band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
                smooth_band_powers = np.mean(band_buffer, axis=0)

                # Compute neurofeedback metrics
                alpha_metric = smooth_band_powers[Band.Alpha] / smooth_band_powers[Band.Delta]
                beta_metric = smooth_band_powers[Band.Beta] / smooth_band_powers[Band.Theta]
                theta_metric = smooth_band_powers[Band.Theta] / smooth_band_powers[Band.Alpha]

                # Append new data to history
                band_history['Time'].append(timestamps[-1])  # Append only the last timestamp
                for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta']):
                    band_history[band].append(smooth_band_powers[i])
                band_history['Alpha Relaxation'].append(alpha_metric)
                band_history['Beta Concentration'].append(beta_metric)
                band_history['Theta Relaxation'].append(theta_metric)

                # Update plot if all arrays have the same length
                if all(len(band_history['Time']) == len(band_history[key]) for key in band_history.keys() if key != 'Time'):
                    for key in lines_bands.keys():
                        lines_bands[key].set_data(band_history['Time'], band_history[key])
                    ax1.relim()
                    ax1.autoscale_view()

                    for key in lines_metrics.keys():
                        lines_metrics[key].set_data(band_history['Time'], band_history[key])
                    ax2.relim()
                    ax2.autoscale_view()

                    plt.pause(0.01)
    
    except KeyboardInterrupt:
        print('Closing!')
    
        plt.ioff()
        plt.savefig(plot_filename)  # Save the plot in the current directory
    
        df = pd.DataFrame(band_history)
        df.to_csv(csv_filename, index=False)  # Save the CSV in the current directory
    
        print(f"Data and plot saved as '{csv_filename}' and '{plot_filename}'.")

        plt.ioff()
        plt.savefig('/mnt/data/eeg_plots.png')

        df = pd.DataFrame(band_history)
        df.to_csv('/mnt/data/eeg_data.csv', index=False)

        print("Data and plots saved.")

