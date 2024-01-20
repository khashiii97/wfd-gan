import os

# Set the directory you want to start from
dir_path = 'ds19/'
monitored_num = 0
unmonitored_num = 0
monitored_sites = {}
# Loop through the directory
for idx, filename in enumerate(os.listdir(dir_path)):
    if '-' in filename:
        monitored_num +=1                       
        site_number = int(filename.split('-')[0])
        monitored_sites[site_number] = monitored_sites.get(site_number, 0) + 1
    else:
        unmonitored_num += 1


print('number of monitored instances: ', monitored_num) #10000
print('number of monitored sites: ', len(monitored_sites.keys()))# 100
print('number of un_monitored instances: ', unmonitored_num)#10000
print('monitored sites individually: ')
print(monitored_sites)