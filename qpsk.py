# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})

# Información binaria que queremos transmitir
binary_info = [1,1,1,0,0,0,0,1,1,0,0,1,1,0]

#Lista con los valores binarios almacenados de dos en dos [[1,1],[1,0],[0,0] ... [1,0]]
combined = [binary_info[i:i + 2] for i in range(0, len(binary_info), 2)] 

samples_per_symbol = 20 # Tamaño en muestras de cada valor binario
symbol_transfer_rate = 2 # Número de símbolos transferidos por segundo
carrier_freq = 5.0

transmission_time = len(combined) / symbol_transfer_rate
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

# Dividimos la informacion binaria en una lista de dos bits (símbolos), ya que tenemos 4 fases

# Calculamos el array de valores combinados
combined_info = []
for x,y in combined:
    combined_info.append(x*2 + y)


combined_info_signal = [] #Vector donde almacenamos los símbolos combinados 
for i in range(0,len(combined_info)):
    # Creamos un vector lleno de tantos 1’s como muestras por símbolo
    all_ones = np.ones(samples_per_symbol)
    # Obtenemos un vector lleno de 0's, 1’s, 2's o 3's dependiendo de binary_info[i]
    one_symbol_values = all_ones*combined_info[i]
    # Concatenamos el símbolo al vector resultante
    combined_info_signal = np.concatenate((combined_info_signal, one_symbol_values))


plt.subplot(5,1,1)
# Dibujamos los valores de la señal binaria combinada
plt.plot(binary_info_signal, label="Binary input (samples)")

# Partimos la grafica en 4 partes y seleccionamos la segunda
plt.subplot(5,1,2)
# Dibujamos los valores de la señal binaria combinada
plt.plot(combined_info_signal, label="Combined binary input (samples)")


# Creamos una linea de tiempo
time = np.linspace(0, (total_samples-1) * sample_time_interval, total_samples)
# Amplitud de 1.0, frecuencia 5 Hrz, fase 0
carrier_signal_zero = 1.0 * np.sin(2 * np.pi * carrier_freq * time + 0)
# Amplitud de 1.0, frecuencia 5 Hrz, fase pi/2
carrier_signal_pi_2 = 1.0 * np.sin(2 * np.pi * carrier_freq * time + 1/2*np.pi)
# Amplitud de 1.0, frecuencia 5 Hrz, fase pi
carrier_signal_pi = 1.0 * np.sin(2 * np.pi * carrier_freq * time + np.pi)
# Amplitud de 1.0, frecuencia 5 Hrz, fase 3/2*pi
carrier_signal_pi_32 = 1.0 * np.sin(2 * np.pi * carrier_freq * time + 3/2*np.pi)

# Creamos la señal resultante después de aplicar QPSK
qpsk_modulated_signal = []

for i in range(0,len(combined_info_signal)):
    
    if (combined_info_signal[i] == 0):
        zero = np.array([carrier_signal_pi_2[i]])
        qpsk_modulated_signal = np.concatenate((qpsk_modulated_signal,zero))

    elif (combined_info_signal[i] == 1):
        one = np.array([carrier_signal_pi[i]])
        qpsk_modulated_signal = np.concatenate((qpsk_modulated_signal,one))

    elif (combined_info_signal[i] == 2):
        two = np.array([carrier_signal_pi_32[i]])
        qpsk_modulated_signal = np.concatenate((qpsk_modulated_signal, two))

    else:
        three = np.array([carrier_signal_zero[i]])
        qpsk_modulated_signal = np.concatenate((qpsk_modulated_signal,three))


# Partimos la grafica en 4 partes y seleccionamos la tercera
plt.subplot(5,1,3)
# Dibujamos los valores de la señal resultante que queremos transmitir
plt.plot(time, qpsk_modulated_signal, c="orange", label="Output in QPSK modulation (sec)")


# Calculamos FFT sobre la señal resultante, y la desplazamos
freq_domain_signal = np.fft.fftshift(np.fft.fft(qpsk_modulated_signal))
# Del FFT obtenemos la parte real (frecuencias)
freq_domain_signal_mag = np.abs(freq_domain_signal) / total_samples
# # Del FFT obtenemos la parte imaginaria (fases)
freq_domain_signal_phase = np.angle(freq_domain_signal)

# Generamos un espacio de frecuencias para un determinado sample rate
freq = np.linspace(-sample_rate/2.0, sample_rate/2.0, total_samples)
# Partimos la grafica en 4 partes y seleccionamos la cuarta
plt.subplot(5,1,4)
# Hacemos visible solo la parte positiva
plt.xlim(0,sample_rate/2.0)
# Multiplicamos por 2 para sumar la amplitud de la parte especular negativa
plt.tight_layout() # Para que se vean los números de las x-axis
plt.plot(freq, freq_domain_signal_mag * 2, "-", c="r", label="Output in frequency domain (Hrz)")


plt.savefig('QPSK.png', dpi=1000.0, bbox_inches='tight', pad_inches=0.5)