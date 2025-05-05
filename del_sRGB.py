from PIL import Image


compiled = 0
converted = 0
errors = []
def fix_image_srgb_profile(file_path):
    global compiled, converted, errors
    try:
        img = Image.open(file_path)
        if img.mode != "RGB":
            print("[Picture Mode Covnert to RGB]: {} Done.".format(file_path))
            img = img.convert('RGB')
            converted += 1
        img.save(file_path, icc_profile=None)
    except Exception as e:
        print("[{}]: in '{}'.".format(e, file_path))
        errors.append("[{}]: in '{}'.".format(e, file_path))
        
    print("[Picture sRGB Profile Deleted]: {}".format(file_path))

import os
current_address = os.path.dirname(os.path.abspath(__file__))
for parent, dirnames, filenames in os.walk(current_address):
     for filename in filenames:
        fix_image_srgb_profile(os.path.join(parent,filename))
        compiled += 1

print("Done.\n=============================================================")
print("Files compiled: {}".format(compiled))
print("Files converted: {}".format(converted))
print("Error File: \n{}".format("\n".join(errors)))
input()
