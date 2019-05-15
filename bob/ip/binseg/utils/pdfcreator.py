#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import os


def get_paths(output_path, filename):
    """
    Parameters
    ----------
    output_path : str
        path in which to look for files
    filename : str

    Returns
    -------
    list 
        list of file paths
    """
    datadir = Path(output_path)
    file_paths = sorted(list(datadir.glob('**/{}'.format(filename))))
    file_paths = [f.as_posix() for f in file_paths]
    return file_paths


def create_pdf(output_path, file_paths, title, tex_filename):
    # setup tex doc
    textitle = "\\section*{{{}}} \n".format(title, 42)
    texinit = "\\documentclass{{article}} \\usepackage[utf8]{{inputenc}} \\usepackage[a4paper, margin=2cm]{{geometry}} \\usepackage{{graphicx}} \\begin{{document}} \n".format(42)
    texclose = "\\end{{document}} \n".format(42)
    with open (os.path.join(output_path,tex_filename), "w+") as outfile:
        outfile.write(texinit)
        outfile.write(textitle)
        for f in file_paths:
            outfile.write("\\includegraphics[width=0.5\\textwidth]{{{}}} \n".format(f,42))
        outfile.write(texclose)
    # create pdf
    os.system("pdflatex -output-directory {} {}".format(output_path, os.path.join(output_path,tex_filename)))