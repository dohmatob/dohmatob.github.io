import sys
import os
import shutil
import re

# Misc
ipynb_filename = sys.argv[1]
filename = os.path.basename(ipynb_filename.replace(".ipynb", ""))
md_filename = "%s.md" % filename
files_dir = "%s_files" % filename

# Use nbconvert to do the initial conversion
os.system("jupyter nbconvert %s --to markdown" % ipynb_filename)

# Humm, not there yet. Fix some stuff
python_code_regex = re.compile("(```python)(.+?)(```)",
                               re.DOTALL or re.MULTILINE)
img_regex = re.compile("(!\[png\])\((%s/%s_6_0.png)(\))" % (files_dir,
                                                            filename))
with open(md_filename, "r") as fd:
    md = fd.read()
    for item in python_code_regex.finditer(md):
        md = md.replace(
            item.group(),
            "{%% highlight python %%}%s{%% endhighlight %%}" % item.group(2))
    for item in img_regex.finditer(md):
        md = md.replace(item.group(),
                        '<img src="/assets/%s"/>' % item.group(2))
    print(md)
with open(md_filename, "w") as fd:
    fd.write(md)

# Final sip
shutil.move(files_dir, "assets")
print("+" * 80)
print("OK! Move %s to your _posts directory and rename it accordingly" % (
    md_filename))
