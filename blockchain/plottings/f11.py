# figure 1

import matplotlib.pyplot as plt
from os import listdir
import sys
import os

log_folder = sys.argv[1]
chosen_device_idx = f"device_{sys.argv[2]}"

all_rounds_log_files = []
list_folders_comm = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
for f in list_folders_comm:
	for file in listdir(f"{log_folder}/{f}"):
		if file.startswith('accuracy'):
			all_rounds_log_files.append(f"{f}/{file}")
			break

draw_comm_rounds = len(all_rounds_log_files)

plt.figure(dpi=250)
f, ax = plt.subplots(1)

# get num of devices with their maliciousness
# benign_devices_idx_list = []
# malicious_devices_idx_list = []
# comm_1_file_path = f"{log_folder}/comm_1/accuracy_comm_1.txt"
# file = open(comm_1_file_path,"r") 
# log_whole_text = file.read() 
# lines_list = log_whole_text.split("\n")
# for line in lines_list:
# 	if line.startswith('device'):
# 		device_idx = line.split(":")[0].split(" ")[0]
# 		device_maliciousness = line.split(":")[0].split(" ")[-1]
# 		if device_maliciousness == 'M':
# 			malicious_devices_idx_list.append(device_idx)
# 		else:
# 			benign_devices_idx_list.append(device_idx)

# total_malicious_devices = len(malicious_devices_idx_list)
# total_devices = len(malicious_devices_idx_list + benign_devices_idx_list)
	
device_accuracies_across_rounds = []

for log_file in all_rounds_log_files:
	file = open(f"{log_folder}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		device_idx = line.split(":")[0].split(" ")[0]
		# if line.startswith('device_1'):
		if device_idx == chosen_device_idx:
			accuracy = round(float(line.split(":")[-1]), 3)
			device_accuracies_across_rounds.append(accuracy)
			break
	file.close()

device_accuracies_across_rounds = device_accuracies_across_rounds[:draw_comm_rounds]
print(f"device_accuracies_across_rounds: {device_accuracies_across_rounds}")

# draw graphs over all available comm rounds
plt.plot(range(draw_comm_rounds), device_accuracies_across_rounds, label=f'Global model accuracy for {chosen_device_idx}', color='orange')

# if device_accuracies_across_rounds:
# 	annotating_points = 5
# 	skipped_1 = False
# 	for accuracy_iter in range(len(device_accuracies_across_rounds)):
# 		if not accuracy_iter % (len(device_accuracies_across_rounds) // annotating_points):
# 			if not skipped_1:
# 				skipped_1 = True
# 				continue
# 			plt.annotate(device_accuracies_across_rounds[accuracy_iter], xy=(accuracy_iter, device_accuracies_across_rounds[accuracy_iter]), size=12)

plt.legend(loc='center', bbox_to_anchor=(0.32,0.7))
ax.set_ylim(ymin=0, ymax=1)
plt.xlabel('Communication Round')
plt.ylabel('Accuracy')
plt.title('Global Model Accuracy')
# plt.show()
fname = f'blockchain/plottings_logs/{log_folder.split("/")[2]}_global_model_accuracy_of_{chosen_device_idx}.pdf'
plt.savefig(fname)
os.system(f"open {fname}")
