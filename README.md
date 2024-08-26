# WiFi-See
OELP PROJECT
Objectives: Develop a WiFi-based human sensing system.
Deliverables: WiFi signature database, deep learning model, Raspberry Pi implementation.
Timeline: July - December 2024
Tools & Technologies: Python, TensorFlow, SciPy, Raspberry Pi, etc.

Using the Rasberry:
1. Setting Up the Raspberry Pi
    a. Gather the Required Hardware
      Raspberry Pi board (preferably Raspberry Pi 4)
      MicroSD card (at least 16GB)
      Power supply
      HDMI cable (if connecting to a monitor)
      USB keyboard and mouse (if needed)
      WiFi adapter (if not using the built-in WiFi on the Raspberry Pi 3/4)
     
    b. Install the Operating System (Raspberry Pi OS)
      1.Download Raspberry Pi Imager:
        Download from the official website: Raspberry PiImager(https://www.raspberrypi.com/software/operating-systems/)
   
      2.Flash the MicroSD Card:
        Insert the MicroSD card into your computer.
        Open Raspberry Pi Imager.
        Select the OS: Choose Raspberry Pi OS (32-bit is recommended).
        Select the SD card: Choose your MicroSD card.
        Click Write to flash the OS onto the card. // copy the downloaded rasberry files into the SD card.
    
      3.First Boot:
        Insert the MicroSD card into the Raspberry Pi.
        Connect the Raspberry Pi to a monitor using the HDMI cable.
        Plug in the keyboard, mouse, and power supply.
        The Raspberry Pi will boot up, and you will see the setup screen. // there you can start installing the os . & give the password for it .
    
      4.Initial Setup:
        Follow the on-screen prompts to set up your language, time zone, and WiFi.
        Set a username and password.
    
      5.Enable SSH (optional but recommended for remote access):
        Go to Preferences -> Raspberry Pi Configuration.
        Under the Interfaces tab, enable SSH.
        You can now access your Raspberry Pi remotely via SSH.

    c. Update the Raspberry Pi
        Open the terminal on the Raspberry Pi and run:
        ( to install all the updates if available)
          sudo apt update
          sudo apt upgrade -y
2. Installing Necessary Software
    a. Python and Libraries
      Python is pre-installed on Raspberry Pi OS, but you may need to install additional libraries:
        sudo apt install python3-pip
        pip3 install numpy scipy matplotlib pandas scikit-learn tensorflow
    b. Install WiFi Monitoring Tools
      For collecting WiFi data (e.g., RSSI, CSI):
        sudo apt install iw
        pip3 install pywifi
      To check your WiFi interface:
        iwconfig
3. Working on the Raspberry Pi
  a. Accessing the Raspberry Pi Remotely
    SSH from a Linux/Mac Terminal:
      ssh pi@<raspberry_pi_ip_address>   //Replace <raspberry_pi_ip_address> with your Raspberry Pi's IP address. You can find it by running hostname -I on the Raspberry Pi.
    SSH from Windows:
      Use a tool like PuTTY. (https://www.putty.org/)
      Enter the IP address and connect.
  b. Transferring Files to Raspberry Pi
    Using SCP (Secure Copy Protocol):
      scp your_file.py pi@<raspberry_pi_ip_address>:~/your_directory/
    Using SFTP (Secure File Transfer Protocol):
    Use an SFTP client like FileZilla to drag and drop files.
  c. Running Python Scripts
      Navigate to the directory where your script is located:
        cd ~/your_directory/
      Run your Python script:
        python3 your_script.py
4. Setting Up WiFi Data Collection
    a. Using PyWiFi to Collect WiFi Data
        The code to collect WiFi data was provided earlier. Here's how to run it on the Raspberry Pi:
          python3 wifi_data_collection.py
       Save the data: Ensure the collected data is saved to a CSV file for future use in training your deep learning models.
    b. Real-Time Monitoring
        Use the Raspberry Pi's built-in WiFi interface to monitor WiFi signals in real-time:
          sudo iw dev wlan0 scan | grep -e signal -e SSID
5. Troubleshooting Common Issues
    a. WiFi Issues
      If you have trouble connecting to WiFi or gathering data, ensure your WiFi interface is up and running:
        sudo ifconfig wlan0 up
      Check the WiFi status:
        iwconfig
   b. Performance Optimization
      Disable unnecessary services to free up resources:
          sudo systemctl disable bluetooth
          sudo systemctl disable hciuart
      Monitor system performance:
          top

6. Additional Tools and Tips
    a. VNC for Remote Desktop
        If you want to access the Raspberry Pi's desktop environment remotely:
        Install VNC:
            sudo apt install realvnc-vnc-server
            Enable VNC in Raspberry Pi Configuration.
                
        
