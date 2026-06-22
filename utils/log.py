from datetime import datetime

def printwrite(filename, *log):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = " ".join(str(i) for i in log)
    msg = f"[{timestamp}] {line}"
    print(msg)
    with open(filename, "a", encoding="utf-8") as file:
        file.write(msg + '\n')
