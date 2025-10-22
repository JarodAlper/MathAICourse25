import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
from google import genai

import tkinter as tk
from tkinter import ttk
import random
import torch
import torch.nn as nn

#Prompt For Claude Sonnet 4.5 within VSCode's Github Copilot:  Geneate a python script that creates an interactive sequence guessing game in a pop-up window using tkinter.  The game should display the first 8 terms of sequence pulled from a dataset and let the user guess the next one.

SEQ_LEN = 8


# Define the same model architecture
class MySequenceNet(nn.Module):
    def __init__(self, input_size=8):
        super(MySequenceNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SequenceGame:
    #User interface stuff
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Guess the Next Number")
        self.root.geometry("600x400")
        self.root.configure(bg="#eef5fa")
        self.root.after(150, lambda: self.root.attributes('-topmost', False))
        self.root.attributes('-topmost', True)

        # ttk style for yellow buttons
        style = ttk.Style()
        # Use a theme that respects custom colors
        if style.theme_use() == 'aqua':
            # switch to clam for better color control on macOS
            try:
                style.theme_use('clam')
            except tk.TclError:
                pass
        style.configure('Yellow.TButton',
                        background='#facc15',
                        foreground='#222222',
                        font=("Arial", 12, 'bold'),
                        borderwidth=1,
                        focusthickness=3,
                        focuscolor='none',
                        padding=(10,4))
        style.map('Yellow.TButton',
                  background=[('active', '#fde047'), ('disabled', '#eab308')])

        self.sequence = []
        self.next_term = None

        header = tk.Label(root, text="Guess the next number", font=("Arial", 16, "bold"), bg="#eef5fa", fg="#203040")
        header.pack(pady=(12,4))

        self.seq_label = tk.Label(root, text="", font=("Consolas", 15), bg="#ffffff", fg="#102030", bd=1, relief="solid", padx=10, pady=6)
        self.seq_label.pack(padx=16, pady=8, fill='x')

        entry_frame = tk.Frame(root, bg="#eef5fa")
        entry_frame.pack(pady=4)
        tk.Label(entry_frame, text="Your guess:", font=("Arial", 12), bg="#eef5fa").pack(side=tk.LEFT, padx=(0,6))
        self.guess_var = tk.StringVar()
        self.entry = tk.Entry(entry_frame, textvariable=self.guess_var, font=("Arial", 12), width=7, justify='center')
        self.entry.pack(side=tk.LEFT)

        self.submit_btn = ttk.Button(root, text="Submit", style='Yellow.TButton', command=self.check_guess)
        self.submit_btn.pack(pady=6)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), bg="#eef5fa")
        self.result_label.pack(pady=4)

        btn_frame = tk.Frame(root, bg="#eef5fa")
        btn_frame.pack(pady=6)
        self.new_btn = ttk.Button(btn_frame, text="New Sequence", style='Yellow.TButton', command=self.new_sequence)
        self.new_btn.pack(side=tk.LEFT, padx=6)
        self.quit_btn = ttk.Button(btn_frame, text="Quit", style='Yellow.TButton', command=root.destroy)
        self.quit_btn.pack(side=tk.LEFT, padx=6)

        self.root.bind('<Return>', lambda _e: self.check_guess())
        self.new_sequence()

    #Choose a random sequence from the dataset
    def generate_sequence(self):
        fib_file = '/Users/jarod/gitwork/playground/data/fib_all.txt'
        with open(fib_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print("Total sequences in dataset:", len(lines))
        random_line = random.choice(lines)
        print("Random sequence selected:", random_line)
        
        # Simple format: just comma-separated integers
        terms_int = [int(t.strip()) for t in random_line.split(',')]
        
        first_terms = terms_int[:8]
        next_term = terms_int[8]
        
        print(f"First terms:", first_terms)
        print(f"Next term:", next_term)
        return first_terms, next_term

    def new_sequence(self):
        self.sequence, self.next_term = self.generate_sequence()
        self.seq_label.config(text=", ".join(map(str, self.sequence)) + ", ?")
        self.result_label.config(text="Guess the next number…", fg="#203040")
        self.guess_var.set("")
        self.entry.config(state='normal')
        self.submit_btn.config(state='normal')
        self.entry.focus_set()

    def check_guess(self):
        print("Guessing for sequence=",self.sequence)


        #Get prediction from my model MySequenceNet stored in models/fib_model.pth 
        model = MySequenceNet(input_size=8)
        model.load_state_dict(torch.load('models/fib_model.pth'))
        model.eval()
        fib_tensor = torch.FloatTensor(self.sequence)

        # Make prediction
        with torch.no_grad():
            my_prediction = model(fib_tensor).item()
            my_rounded_prediction = round(my_prediction)

        #Get Gemini's guess
        prompt = f"Given the sequence {self.sequence}, what is the next number?  Give only the number."
        client = genai.Client()
        gemini = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        print("gemini=",gemini.text)
        print("correct answer=",self.next_term)

        if self.submit_btn.instate(['disabled']):
            return
        guess_text = self.guess_var.get().strip()
        try:
            guess_val = int(guess_text)
        except ValueError:
            self.result_label.config(text="Enter a valid integer.", fg="#b91c1c")
            return
        
        response = f"Your guess: {guess_val}\n"
        response += f"MySequenceNet (trained): {my_rounded_prediction}\n"
        response += f"Gemini 2.0 (pretrained): {gemini.text}\n"
        response += f"Correct answer: {self.next_term}"
        self.result_label.config(text=response, fg="#b91c1c")

        self.entry.config(state='disabled')
        self.submit_btn.state(['disabled'])


def main():
    #initialize Gemini API
    root = tk.Tk()
    SequenceGame(root)
    root.mainloop()

if __name__ == '__main__':
    main()
