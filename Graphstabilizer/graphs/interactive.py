#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:56:28 2023

@author: jarn
"""

import tkinter as tk


def Interactivegraphplot(Graph):
    '''
    '''
    pass
window = tk.Tk()

window.title("GraphVisualizer")

def handle_button_press(event):
    window.destroy()


button = tk.Button(text="Close.")
button.bind("Destroy", handle_button_press)
button.pack()

# Start the event loop.
window.mainloop()