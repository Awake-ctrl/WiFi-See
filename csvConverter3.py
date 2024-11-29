import time
import sys
import os

def process_file(input_file, output_file,signals):
    """
    Reads the input text file, filters lines with "CSI_DATA",
    and appends them to the output CSV with timestamps.
    """
    with open(input_file, "r", encoding="utf-16", errors="ignore") as infile, \
         open(output_file, "w") as outfile:
        
        # outfile.write("CSI_DATA_Line,Timestamp\n")  # Header for the CSV
        count=0
        for line in infile:
            
                    if "CSI_DATA" in line:
                        count+=1
                        
                        # timestamped_line = f"{line.strip()},{time.time()}"
                        
                        # outfile.write(timestamped_line + "\n")
                        if (count>100 or count==1):
                            if count<=38000:
                                signal="signal"
                                signal_value=signals
                        
                                timestamped_line = f"{line.strip()}"
                                
                                if (timestamped_line.split(",")[-1]=="CSI_DATA"):
                                    timestamped_line = f"{line.strip()},{signal}"
                                else:
                                    timestamped_line = f"{line.strip()},{signal_value}"
                                outfile.write(timestamped_line + "\n")
                        
                        
           
                    

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Usage : python csvConverter3.py <file.txt> <file.csv> True/False")
        exit(0)
    txt_file=sys.argv[1]
    csv_file=txt_file.split(".")[0]+".csv"
    csv_file=sys.argv[2]
    signal=sys.argv[3]
    true=["T","t","True","true","1"]
    false=["F","f","False","false","0"]
    if signal in true:
        signal=True
    elif signal in false:
        signal=False
    if not os.path.isfile(txt_file):
        print(f"Error:{txt_file} does not exist.")
        sys.exit(1)
    
    process_file(txt_file,csv_file,signal)
