
from matplotlib import pyplot as plt

def performance_analysis_tegrastats_gpu_XAVIER(file_name):
  plt.rcParams.update({'font.size': 15})
  #Looking for GPU Memory Consumption.
  word = 'GR3D_FREQ'
  mem, freq = [], []
  index, zero_mem = 0, 0
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      index = 0
      for line in lines:
          # check if string present on a current line
          if line.find(word) != -1:
              section_of_interest = line[line.find(word)+9:line.find(word)+16]
              #print("GR3D_FREQ:",section_of_interest)
              if (int(section_of_interest[section_of_interest.find(" "):section_of_interest.find("%@")]) == 0):
                zero_mem = zero_mem + 1
              else:
                mem.append(int(section_of_interest[section_of_interest.find(" "):section_of_interest.find("%@")]))
              #print("Memory:",mem)
              freq.append(int(section_of_interest[section_of_interest.find("%@")+2:]))
              #print("Frequency:",freq)
              index = index + 1

  print("---------- GPU Utilization ----------")   
  print("Average GPU (%) Memory Consumption:", sum(mem)/len(mem))    
  print("Ratio of Time GPU had no Load (%):", 100*zero_mem/len(lines))
  print("Average Frequency:", sum(freq)/len(freq))

  plt.figure(figsize=(15, 6))
  plt.hist(mem, bins=range(min(mem), max(mem) + 1, 1), color="mediumblue")
  plt.legend()
  plt.xlabel('GPU Utilization (%)')
  plt.ylabel('Samples (T=100ms)')
  plt.grid()
  plt.title("Jetson Xavier NX - GPU Utilization (%)")
  plt.show()

  plt.figure(figsize=(15, 6))
  plt.hist(freq, bins=range(min(freq), max(freq) + 10, 10), color="crimson")
  plt.legend()
  plt.xlabel('GPU Frequency (MHz)')
  plt.ylabel('Samples (T=100ms)')
  plt.title("Jetson Xavier NX - GPU Frequency (MHz)")
  plt.grid()
  plt.show()

  #Looking for CPU Workload
  word = "CPU"
  cpu1, cpu2, cpu3, cpu4, cpu5, cpu6 = [],[],[],[],[],[]
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      for line in lines:
          # check if string present on a current line
          if line.find(word) != -1:
              section_of_interest = line[line.find(word)+3:line.find(word)+60]
              #print("CPU:",section_of_interest)
              #print("CPU 1:",section_of_interest[section_of_interest.find("[")+1:section_of_interest.find(",")])
              sect = section_of_interest[section_of_interest.find("[")+1:section_of_interest.find(",")]
              cpu1.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 2:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu2.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 3:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu3.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 4:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu4.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 5:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu5.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 6:",section_of_interest[:section_of_interest.find("]")])
              sect = section_of_interest[:section_of_interest.find("]")]
              cpu6.append(int(sect[:sect.find("%@")]))
  
  print("---------- CPU Utilization ----------")  
  print("Average CPU 1 (%) Workload:", sum(cpu1)/len(cpu1))    
  print("Average CPU 2 (%) Workload:", sum(cpu2)/len(cpu2))  
  print("Average CPU 3 (%) Workload:", sum(cpu3)/len(cpu3))  
  print("Average CPU 4 (%) Workload:", sum(cpu4)/len(cpu4))  
  print("Average CPU 5 (%) Workload:", sum(cpu5)/len(cpu5))  
  print("Average CPU 6 (%) Workload:", sum(cpu6)/len(cpu6))  

  plt.style.use('seaborn-deep')
  plt.figure(figsize=(15, 6))
  plt.hist([cpu1,cpu2,cpu3,cpu4,cpu5,cpu6], alpha=0.8, label=['cpu1', 'cpu2','cpu3', 'cpu4','cpu5', 'cpu6'])
  plt.legend()
  plt.xlabel('CPU Utilization (%)')
  plt.ylabel('Samples (T=100ms)')
  plt.grid()
  plt.title("Jetson Xavier NX - CPU Utilization (%)")
  plt.show()

  #Looking for Power Consumption.
  words = ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]
  vdd_in, vdd_cpu_gpu, vdd_soc = [], [], []
  vdd_in_current, vdd_cpu_gpu_current, vdd_soc_current = 0,0,0
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      for line in lines:
          # check if string present on a current line
          if line.find(words[0]) != -1:
              section_of_interest = line[line.find(words[0])+7:line.find(words[0])+20]
              #print("VDD_IN:",section_of_interest)
              #print("Power (mw):",section_of_interest[section_of_interest.find("/")+1:section_of_interest.find(" ")])
              vdd_in.append(int(section_of_interest[section_of_interest.find("/")+1:section_of_interest.find(" ")]))
              vdd_in_current = vdd_in_current + int(section_of_interest[:section_of_interest.find("/")])
          if line.find(words[1]) != -1:
              section_of_interest = line[line.find(words[1])+16:line.find(words[1])+30]
              #print("VDD_CPU_GPU_CV:",section_of_interest)
              #print("Power (mw):",section_of_interest[section_of_interest.find("/")+1:section_of_interest.find(" ")])
              vdd_cpu_gpu.append(int(section_of_interest[section_of_interest.find("/")+1:section_of_interest.find(" ")]))
              vdd_cpu_gpu_current = vdd_cpu_gpu_current + int(section_of_interest[:section_of_interest.find("/")])
          if line.find(words[2]) != -1:
              section_of_interest = line[line.find(words[2])+7:line.find(words[2])+30]
              #print("VDD_SOC:",section_of_interest)
              #print("Power (mw):",section_of_interest[section_of_interest.find("/")+1:])
              vdd_soc.append(int(section_of_interest[section_of_interest.find("/")+1:]))
              vdd_soc_current = vdd_soc_current + int(section_of_interest[:section_of_interest.find("/")])
  
  print("---------- POWER CONSUMPTION ----------")    
  print("Accumulate Average VDD_IN (mw):", sum(vdd_in)/len(vdd_in))    
  print("Accumulate Average VDD_CPU_GPU_CV (mw):", sum(vdd_cpu_gpu)/len(vdd_cpu_gpu))
  print("Accumulate Average VDD_SOC (mw):", sum(vdd_soc)/len(vdd_soc))
  print("--------------------") 
  print("Total Average VDD_IN (mw):", vdd_in_current/len(lines))    
  print("Total Average VDD_CPU_GPU_CV (mw):", vdd_cpu_gpu_current/len(lines))
  print("Total Average VDD_SOC (mw):", vdd_soc_current/len(lines))
  
  plt.figure(figsize=(15, 6))
  x = range(len(vdd_in))
  plt.scatter(x,vdd_in, color="crimson")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Power (mw)')
  plt.grid()
  plt.title("Jetson Xavier NX - Average Power IN (mw)")
  plt.show()

  plt.figure(figsize=(15, 6))
  x = range(len(vdd_cpu_gpu))
  plt.scatter(x,vdd_cpu_gpu, color="green")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.title("Jetson Xavier NX - Average Power (mw)")
  plt.grid()
  plt.show()

  plt.figure(figsize=(15, 6))
  x = range(len(vdd_soc))
  plt.scatter(x,vdd_soc, color="gold")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Power (mw)')
  plt.title("Jetson Xavier NX - Average Power (mw)")
  plt.grid()
  plt.show()

  #Looking for GPU and CPU Temp.
  words = ["GPU@", "CPU@"]
  gpu_temp, cpu_temp= [], []
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      index = 0
      for line in lines:
          # check if string present on a current line
          if line.find(word) != -1:
              section_of_interest = line[line.find(words[0])+4:line.find(words[0])+8]
              #print("gpu_temp:",section_of_interest)
              gpu_temp.append(int(section_of_interest[:2]))
              section_of_interest = line[line.find(words[1])+4:line.find(words[1])+8]
              #print("cpu_temp:",section_of_interest)
              cpu_temp.append(int(section_of_interest[:2]))

  print("---------- GPU and CPU Temperature ----------")   
  print("Average GPU Temp:", sum(gpu_temp)/len(gpu_temp))    
  print("Average CPU Temp:", sum(cpu_temp)/len(cpu_temp))

  plt.figure(figsize=(15, 6))
  x = range(len(gpu_temp))
  plt.scatter(x,gpu_temp, color="green")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Temperature (°C)')
  plt.grid()
  plt.show()

  plt.figure(figsize=(15, 6))
  x = range(len(cpu_temp))
  plt.scatter(x,cpu_temp, color="gold")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Temperature (°C)')
  plt.grid()
  plt.show()

  return 0

def performance_analysis_tegrastats_gpu_NANO(file_name):
  #Looking for GPU Memory Consumption.
  word = 'GR3D_FREQ'
  mem, freq = [], []
  index, zero_mem = 0, 0
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      index = 0
      for line in lines:
          # check if string present on a current line
          if line.find(word) != -1:
              section_of_interest = line[line.find(word)+9:line.find(word)+16]
              #print("GR3D_FREQ:",section_of_interest)
              if (int(section_of_interest[section_of_interest.find(" "):section_of_interest.find("%@")]) == 0):
                zero_mem = zero_mem + 1
              else:
                mem.append(int(section_of_interest[section_of_interest.find(" "):section_of_interest.find("%@")]))
              #print("Memory:",mem)
              freq.append(int(section_of_interest[section_of_interest.find("%@")+2:]))
              #print("Frequency:",freq)
              index = index + 1
  
  print("---------- GPU Utilization ----------")    
  print("Average GPU (%) Memory Consumption:", sum(mem)/len(mem))    
  print("Ratio of Time GPU had no Load (%):", 100*zero_mem/len(lines))
  print("Average Frequency:", sum(freq)/len(freq))

  plt.figure(figsize=(15, 6))
  plt.hist(mem, bins=range(min(mem), max(mem) + 1, 1), color="mediumblue")
  plt.legend()
  plt.xlabel('GPU Utilization (%)')
  plt.ylabel('Samples (T=100ms)')
  plt.grid()
  plt.title("Jetson Nano - GPU Utilization (%)")
  plt.show()

  plt.figure(figsize=(15, 6))
  plt.hist(freq, bins=range(min(freq), max(freq) + 10, 10), color="crimson")
  plt.legend()
  plt.xlabel('GPU Frequency (MHz)')
  plt.ylabel('Samples (T=100ms)')
  plt.grid()
  plt.title("Jetson Nano - GPU Frequency (MHz)")
  plt.show()

  #Looking for CPU Workload
  word = "CPU"
  cpu1, cpu2, cpu3, cpu4, cpu5, cpu6 = [],[],[],[],[],[]
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      for line in lines:
          # check if string present on a current line
          if line.find(word) != -1:
              section_of_interest = line[line.find(word)+3:line.find(word)+60]
              #print("CPU:",section_of_interest)
              #print("CPU 1:",section_of_interest[section_of_interest.find("[")+1:section_of_interest.find(",")])
              sect = section_of_interest[section_of_interest.find("[")+1:section_of_interest.find(",")]
              cpu1.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 2:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu2.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 3:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu3.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 4:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu4.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 5:",section_of_interest[:section_of_interest.find(",")])
              sect = section_of_interest[:section_of_interest.find(",")]
              cpu5.append(int(sect[:sect.find("%@")]))
              section_of_interest = section_of_interest[section_of_interest.find(",")+1:]
              #print("CPU 6:",section_of_interest[:section_of_interest.find("]")])
              sect = section_of_interest[:section_of_interest.find("]")]
              cpu6.append(int(sect[:sect.find("%@")]))
  
  print("---------- CPU Utilization ----------")  
  print("Average CPU 1 (%) Workload:", sum(cpu1)/len(cpu1))    
  print("Average CPU 2 (%) Workload:", sum(cpu2)/len(cpu2))  
  print("Average CPU 3 (%) Workload:", sum(cpu3)/len(cpu3))  
  print("Average CPU 4 (%) Workload:", sum(cpu4)/len(cpu4))  
  print("Average CPU 5 (%) Workload:", sum(cpu5)/len(cpu5))  
  print("Average CPU 6 (%) Workload:", sum(cpu6)/len(cpu6))  

  plt.style.use('seaborn-deep')
  plt.figure(figsize=(15, 6))
  plt.hist([cpu1,cpu2,cpu3,cpu4,cpu5,cpu6], alpha=0.8, label=['cpu1', 'cpu2','cpu3', 'cpu4','cpu5', 'cpu6'])
  plt.legend()
  plt.xlabel('CPU Utilization (%)')
  plt.ylabel('Samples (T=100ms)')
  plt.grid()
  plt.title("Jetson Nano - CPU Utilization (%)")
  plt.show()

  #Looking for Power Consumption.
  words = ["POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"]
  vdd_in, vdd_cpu_gpu, vdd_soc = [], [], []
  vdd_in_current, vdd_cpu_gpu_current, vdd_soc_current = 0,0,0
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      for line in lines:
          # check if string present on a current line
          if line.find(words[0]) != -1:
              section_of_interest = line[line.find(words[0])+9:line.find(words[0])+30]
              #print("POM_5V_IN:",section_of_interest)
              #print("Power (mw):",section_of_interest[section_of_interest.find("/")+1:section_of_interest.find("P")-1])
              vdd_in.append(int(section_of_interest[section_of_interest.find("/")+1:section_of_interest.find("P")-1]))
              vdd_in_current = vdd_in_current + int(section_of_interest[:section_of_interest.find("/")])
          if line.find(words[1]) != -1:
              section_of_interest = line[line.find(words[1])+10:line.find(words[1])+30]
              #print("POM_5V_GPU:",section_of_interest)
              #print("Power (mw):",section_of_interest[section_of_interest.find("/")+1:section_of_interest.find("P")-1])
              vdd_cpu_gpu.append(int(section_of_interest[section_of_interest.find("/")+1:section_of_interest.find("P")-1]))
              vdd_cpu_gpu_current = vdd_cpu_gpu_current + int(section_of_interest[:section_of_interest.find("/")])
          if line.find(words[2]) != -1:
              section_of_interest = line[line.find(words[2])+10:line.find(words[2])+30]
              #print("POM_5V_CPU:",section_of_interest)
              #print("Power (mw):",section_of_interest[section_of_interest.find("/")+1:])
              vdd_soc.append(int(section_of_interest[section_of_interest.find("/")+1:]))
              vdd_soc_current = vdd_soc_current + int(section_of_interest[:section_of_interest.find("/")])

  print("---------- POWER CONSUMPTION ----------")    
  print("Average POM_5V_IN:", sum(vdd_in)/len(vdd_in))    
  print("Average POM_5V_GPU:", sum(vdd_cpu_gpu)/len(vdd_cpu_gpu))
  print("Average POM_5V_CPU:", sum(vdd_soc)/len(vdd_soc))
  print("--------------------")  
  print("Total Average VDD_IN (mw):", vdd_in_current/len(lines))    
  print("Total Average VDD_CPU_GPU_CV (mw):", vdd_cpu_gpu_current/len(lines))
  print("Total Average VDD_SOC (mw):", vdd_soc_current/len(lines))

  plt.figure(figsize=(15, 6))
  x = range(len(vdd_in))
  plt.scatter(x,vdd_in, color="crimson")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Power (mw)')
  plt.grid()
  plt.title("Jetson Nano - Average Power IN (mw)")
  plt.show()

  plt.figure(figsize=(15, 6))
  x = range(len(vdd_cpu_gpu))
  plt.scatter(x,vdd_cpu_gpu, color="green")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Power (mw)')
  plt.grid()
  plt.show()

  plt.figure(figsize=(15, 6))
  x = range(len(vdd_soc))
  plt.scatter(x,vdd_soc, color="gold")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Power (mw)')
  plt.grid()
  plt.show()

    #Looking for GPU and CPU Temp.
  words = ["GPU@", "CPU@"]
  gpu_temp, cpu_temp= [], []
  with open(file_name, 'r') as fp:
      # read all lines in a list
      lines = fp.readlines()
      index = 0
      for line in lines:
          # check if string present on a current line
          if line.find(word) != -1:
              section_of_interest = line[line.find(words[0])+4:line.find(words[0])+8]
              #print("gpu_temp:",section_of_interest)
              gpu_temp.append(int(section_of_interest[:2]))
              section_of_interest = line[line.find(words[1])+4:line.find(words[1])+8]
              #print("cpu_temp:",section_of_interest)
              cpu_temp.append(int(section_of_interest[:2]))

  print("---------- GPU and CPU Temperature ----------")   
  print("Average GPU Temp:", sum(gpu_temp)/len(gpu_temp))    
  print("Average CPU Temp:", sum(cpu_temp)/len(cpu_temp))

  plt.figure(figsize=(15, 6))
  x = range(len(gpu_temp))
  plt.scatter(x,gpu_temp, color="green")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Temperature (°C)')
  plt.grid()
  plt.show()

  plt.figure(figsize=(15, 6))
  x = range(len(cpu_temp))
  plt.scatter(x,cpu_temp, color="gold")
  plt.legend()
  plt.xlabel('Samples (T=100ms)')
  plt.ylabel('Average Temperature (°C)')
  plt.grid()
  plt.show()

  return 0

"""XAVIER NX **ANALYSIS**"""

file_name = r'video_22_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

file_name = r'video_37_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

file_name = r'video_38_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

file_name = r'video_44_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

file_name = r'video_45_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

file_name = r'video_59_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

file_name = r'video_60_log.txt'
performance_analysis_tegrastats_gpu_XAVIER(file_name)

"""NANO **ANALYSIS**"""

file_name = r'video_22_log.txt'
performance_analysis_tegrastats_gpu_NANO(file_name)

file_name = r'video_37_log.txt'
performance_analysis_tegrastats_gpu_NANO(file_name)

file_name = r'video_45_log.txt'
performance_analysis_tegrastats_gpu_NANO(file_name)