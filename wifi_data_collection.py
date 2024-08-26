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

def collect_data(num_scans, interface=0):
    """ Collect WiFi data over a specified number of scans. """
    data = []
    for i in range(num_scans):
        print(f"Scan {i + 1}/{num_scans}")
        networks = scan_wifi(interface)
        for ssid, signal in networks:
            data.append((ssid, signal))
        time.sleep(5)  # Wait before the next scan
    return data

def save_data(data, filename="wifi_data.csv"):
    """ Save the collected data to a CSV file. """
    df = pd.DataFrame(data, columns=["SSID", "Signal_Strength"])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    num_scans = 10  # Number of scans to perform
    data = collect_data(num_scans)
    save_data(data)

if __name__ == "__main__":
    main()
