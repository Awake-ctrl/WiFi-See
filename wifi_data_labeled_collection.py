import pywifi
import time
import pandas as pd

def scan_wifi(interface):
    """ Scan for WiFi networks and return a list of SSIDs and signal strengths. """
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[interface]
    iface.scan()
    time.sleep(2)  # Wait for scan results to be available
    scan_results = iface.scan_results()
    networks = []
    for network in scan_results:
        networks.append((network.ssid, network.signal))
    return networks

def collect_labeled_data(num_scans, label, interface=0):
    """ Collect labeled WiFi data over a specified number of scans. """
    data = []
    for i in range(num_scans):
        print(f"Scan {i + 1}/{num_scans} with label: {label}")
        networks = scan_wifi(interface)
        for ssid, signal in networks:
            data.append((ssid, signal, label))
        time.sleep(5)  # Wait before the next scan
    return data

def save_labeled_data(data, filename="wifi_data_labeled.csv"):
    """ Save the collected labeled data to a CSV file. """
    df = pd.DataFrame(data, columns=["SSID", "Signal_Strength", "Label"])
    df.to_csv(filename, index=False)
    print(f"Labeled data saved to {filename}")

def main():
    num_scans = 10  # Number of scans to perform
    label = "human_present"  # Change to "no_human" for no human presence
    data = collect_labeled_data(num_scans, label)
    save_labeled_data(data)

if __name__ == "__main__":
    main()
