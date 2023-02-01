# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})

# Información binaria que queremos transmitir
binary_info = [1,0,1,0,1,0,0,1,1,0,0,1,1,0,1]
samples_per_symbol = 20 # Tamaño en muestras de cada valor binario
symbol_transfer_rate = 2 # Number of symbols transferred in a second
carrier_freq = 4 # Frecuencia de la portadora alta en Hz

transmission_time = len(binary_info) / symbol_transfer_rate
sample_rate = samples_per_symbol * symbol_transfer_rate
total_samples = int(sample_rate * transmission_time)
sample_time_interval = 1.0 / sample_rate

binary_info_signal = [] # Vector donde guardaremos la señal binaria resultante

for i in range(0,len(binary_info)):
    # Creamos un vector lleno de tantos 1’s como muestras por símbolo
    all_ones = np.ones(samples_per_symbol)
    # Obtenemos un vector lleno de 0's o 1’s dependiendo de binary_info[i]
    one_symbol_values = all_ones*binary_info[i]
    # Concatenamos el símbolo al vector resultante
    binary_info_signal = np.concatenate((binary_info_signal, one_symbol_values))

# Partimos la grafica en 4 partes y seleccionamos la primera
plt.subplot(4,1,1)
# Dibujamos los valores de la señal binaria
plt.plot(binary_info_signal, label="Binary input (samples)")

# Creamos una linea de tiempo
time = np.linspace(0, (total_samples-1) * sample_time_interval, total_samples)
# Amplitud de 1.0, frecuencia 4 Hrz, fase 0
carrier_signal = 1.0 * np.sin(2 * np.pi * carrier_freq * time + 0)

# Partimos la grafica en 4 partes y seleccionamos la segunda
plt.subplot(4,1,2)
# Dibujamos los valores de la señal portadora alta frequencia
plt.plot(time, carrier_signal, c="orange", label=str(carrier_freq) + "Hz Carrier signal (sec)")

# Creamos la señal resultante después de aplicar ASK, modulación binaria por amplitud
ask_modulated_signal = carrier_signal * binary_info_signal

# Partimos la grafica en 4 partes y seleccionamos la tercera
plt.subplot(4,1,3)
# Dibujamos los valores de la señal resultante que queremos transmitir
plt.plot(time, ask_modulated_signal, c="g", label="Output in ASK modulation (sec)")

# Calculamos FFT sobre la señal resultante, y la desplazamos (ver PySDR para saber por qué)
freq_domain_signal = np.fft.fftshift(np.fft.fft(ask_modulated_signal))
# Del FFT obtenemos la parte real (frecuencias)
freq_domain_signal_mag = np.abs(freq_domain_signal) / total_samples
# # Del FFT obtenemos la parte imaginaria (fases)
freq_domain_signal_phase = np.angle(freq_domain_signal)

#Generamos un espacio de frecuencias para un determinado sample rate
freq = np.linspace(-sample_rate/2.0, sample_rate/2.0, total_samples )
# Partimos la grafica en 4 partes y seleccionamos la cuarta
plt.subplot(4,1,4)
# Hacemos visible solo la parte positiva
plt.xlim(0,sample_rate/2.0)
# Multiplicamos por 2 para sumar la amplitud de la parte especular negativa
plt.tight_layout() # Para que se vean los números de las x-axis
plt.plot(freq, freq_domain_signal_mag * 2, "-", c="r", label="Output in frequency domain (Hrz)")

#Guardamos la gráfica de la onda modulada
plt.savefig('ASK.png', dpi=1000.0, bbox_inches='tight', pad_inches=0.5)