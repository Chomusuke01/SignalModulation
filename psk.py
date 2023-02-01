# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})

# Información binaria que queremos transmitir
binary_info = [1,0,1,0,1,0,0,1,1,0,0,1,1,0,1]
samples_per_symbol = 20 # Tamaño en muestras de cada valor binario
symbol_transfer_rate = 2 # Número de símbolos transferidos por segundo
carrier_freq = 4.0

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
plt.subplot(5,1,1)
# Dibujamos los valores de la señal binaria
plt.plot(binary_info_signal, label="Binary input (samples)")

# Creamos una linea de tiempo
time = np.linspace(0, (total_samples-1) * sample_time_interval, total_samples)
# Amplitud de 1.0, frecuencia 4 Hrz, fase 0
carrier_signal_zero = 1.0 * np.sin(2 * np.pi * carrier_freq * time + 0)
# Amplitud de 1.0, frecuencia 4 Hrz, fase pi
carrier_signal_pi = 1.0 * np.sin(2 * np.pi * carrier_freq * time + np.pi)

#Partimos la grafica en 5 partes y nos quedamos con la segunda
plt.subplot(5,1,2)
# Dibujamos los valores de la señal portadora con fase 0
plt.plot(time, carrier_signal_zero, c="orange", label="0 radians phase")
# Partimos la grafica en 5 partes y seleccionamos la tercera
plt.subplot(5,1,3)
# Dibujamos los valores de la señal portadora alta frequencia
plt.plot(time, carrier_signal_pi, c="orange", label=str(np.pi) + "radians phase")


# Creamos la señal resultante después de aplicar FSK, modulación binaria por frecuencia
psk_modulated_signal = []
for i in range(0,len(binary_info_signal)):

    if (binary_info_signal[i] == 0):
        zero = np.array([carrier_signal_zero[i]])
        psk_modulated_signal = np.concatenate((psk_modulated_signal,zero))
    else:
        one = np.array([carrier_signal_pi[i]])
        psk_modulated_signal = np.concatenate((psk_modulated_signal,one))


# Partimos la grafica en 5 partes y seleccionamos la cuarta
plt.subplot(5,1,4)
# Dibujamos los valores de la señal resultante que queremos transmitir
plt.plot(time, psk_modulated_signal, c="g", label="Output in PSK modulation (sec)")


# Calculamos FFT sobre la señal resultante, y la desplazamos
freq_domain_signal = np.fft.fftshift(np.fft.fft(psk_modulated_signal))
# Del FFT obtenemos la parte real (frecuencias)
freq_domain_signal_mag = np.abs(freq_domain_signal) / total_samples
# # Del FFT obtenemos la parte imaginaria (fases)
freq_domain_signal_phase = np.angle(freq_domain_signal)

# Generamos un espacio de frecuencias para un determinado sample rate
freq = np.linspace(-sample_rate/2.0, sample_rate/2.0, total_samples )
# Partimos la grafica en 5 partes y seleccionamos la quinta
plt.subplot(5,1,5)
# Hacemos visible solo la parte positiva
plt.xlim(0,sample_rate/2.0)
# Multiplicamos por 2 para sumar la amplitud de la parte especular negativa
plt.tight_layout() # Para que se vean los números de las x-axis
plt.plot(freq, freq_domain_signal_mag * 2, "-", c="r", label="Output in frequency domain (Hrz)")

plt.savefig('PSK.png', dpi=1000.0, bbox_inches='tight', pad_inches=0.5)