import argparse

parser = argparse.ArgumentParser()
# Use either src_dir or img_list, not both.
parser.add_argument("log", type=str, help="resultMatrix.log file")
args = parser.parse_args()
log_file = args.log

print "======================== Reformating resultMatrix Log File ========================"
f = open(log_file,"r+")
data = f.readlines()
f.seek(0)
for line in data:
    if line.startswith("Model") or line.startswith(" Average") or line.startswith("overall") or line.startswith("----"):
        f.write(line)
f.truncate()
f.close()
