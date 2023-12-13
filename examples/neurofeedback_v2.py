import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import utils
import pandas as pd

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

band_history = {'Time': [], 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': []}
plt.ion()

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

    fig, ax = plt.subplots()
    lines = {band: ax.plot([], [], label=band)[0] for band in ['Delta', 'Theta', 'Alpha', 'Beta']}
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')

    print('Press Ctrl-C in the console to break the while loop.')
    try:
        while True:
            # Acquire data
            eeg_data, timestamps = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            if len(eeg_data) > 0:
                # Process the EEG data
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
                eeg_buffer, filter_state = utils.update_buffer(eeg_buffer, ch_data, notch=True, filter_state=filter_state)

                data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
                band_powers = utils.compute_band_powers(data_epoch, fs)
                band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
                smooth_band_powers = np.mean(band_buffer, axis=0)

                # Update band history for each timestamp
                for timestamp in timestamps:
                    band_history['Time'].append(timestamp)
                    for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta']):
                        band_history[band].append(smooth_band_powers[i])

                # Update plot
                if len(band_history['Time']) == len(band_history['Delta']):
                    for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
                        x_data = band_history['Time']
                        y_data = band_history[band]
                        lines[band].set_data(x_data, y_data)
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.01)


    except KeyboardInterrupt:
        print('Closing!')

        plt.ioff()
        plt.savefig('/mnt/data/eeg_band_powers.png')

        df = pd.DataFrame(band_history)
        df.to_csv('/mnt/data/eeg_band_powers.csv', index=False)

        print("Data and plot saved.")
