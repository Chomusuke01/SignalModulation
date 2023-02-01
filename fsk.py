# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})


# Información binaria que queremos transmitir
binary_info = [1,0,1,0,1,0,0,1,1,0,0,1,1,0,1]
samples_per_symbol = 20 # Tamaño en muestras de cada valor binario
symbol_transfer_rate = 2 # Número de símbolos transferidos por segundo
carrier_freq_hi = 10.0 # Frecuencia alta de la portadora en Hz
carrier_freq_lo = 5.0 # Frecuencia baja de la portadora en Hz

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
# Amplitud de 1.0, frecuencia 5 Hrz, fase 0
carrier_signal_lo = 1.0 * np.sin(2 * np.pi * carrier_freq_lo * time + 0)
# Amplitud de 1.0, frecuencia 10 Hrz, fase 0
carrier_signal_hi = 1.0 * np.sin(2 * np.pi * carrier_freq_hi * time + 0)

#Partimos la grafica en 5 partes y nos quedamos con la segunda
plt.subplot(5,1,2)
# Dibujamos los valores de la señal portadora baja frequencia
plt.plot(time, carrier_signal_lo, c="orange", label=str(carrier_freq_lo) + "Hz Carrier signal (sec)")
# Partimos la grafica en 5 partes y seleccionamos la tercera
plt.subplot(5,1,3)
# Dibujamos los valores de la señal portadora alta frequencia
plt.plot(time, carrier_signal_hi, c="orange", label=str(carrier_freq_hi) + "Hz Carrier signal (sec)")

# Creamos la señal resultante después de aplicar FSK, modulación binaria por frecuencia
fsk_modulated_signal = []
for i in range(0,len(binary_info_signal)):

    if (binary_info_signal[i] == 0):
        zero = np.array([carrier_signal_lo[i]])
        fsk_modulated_signal = np.concatenate((fsk_modulated_signal,zero))
    else:
        one = np.array([carrier_signal_hi[i]])
        fsk_modulated_signal = np.concatenate((fsk_modulated_signal,one))

# Partimos la grafica en 5 partes y seleccionamos la cuarta
plt.subplot(5,1,4)
# Dibujamos los valores de la señal resultante que queremos transmitir
plt.plot(time, fsk_modulated_signal, c="g", label="Output in FSK modulation (sec)")

# Calculamos FFT sobre la señal resultante, y la desplazamos (ver PySDR para saber por qué)
freq_domain_signal = np.fft.fftshift(np.fft.fft(fsk_modulated_signal))
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

plt.savefig('FSK.png', dpi=1000.0, bbox_inches='tight', pad_inches=0.5)