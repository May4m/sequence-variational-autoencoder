
import numpy as np
import matplotlib.pyplot as plt



def sin_signal_freq(shape=[500, 309, 2]):
    """
    Synthetic data
    Generates a list of sine signals with frequency and phase variation
    """

    out = []
    freq_list = []
    for i in range(shape[0]):

        freq = np.random.choice(np.arange(1, 6))
        phase = np.random.choice([1, 4])
        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.sin(freq * x + phase)

        t = np.zeros([shape[1], shape[2]])
        noise = np.random.randn(*x.shape) * 0.1
        t[:, 0] = x + noise
        t[:, 1] = y + noise
        out.append(t), freq_list.append(freq)
    return np.reshape(np.asarray(out), shape), freq_list


def draw_latent_frequency(encoder=None):
    """
    Plots the latent space to show the model has learned to separate different signals variations
    """
    
    color_id = {1: 'red', 2: 'green', 3:'blue', 4: 'orange', 5: 'purple'}

    if encoder is None:
        encoder = load_model('encoder.h5')
    linear_1, amp_1 = sin_signal_freq()
    
    sin_latents_1 = encoder.predict(linear_1)    
    plt.cla();plt.clf()
    plt.scatter(sin_latents_1[:, 0], sin_latents_1[:, 1], s=3.5, color=[color_id[i] for i in amp_1])
    plt.savefig('frequency_with_phase_latents.png', dpi=200)

dataset, _ = sin_signal_freq()
np.random.shuffle(dataset)