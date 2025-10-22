import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
from google import genai

import tkinter as tk
from tkinter import ttk
import random

# Interactive sequence guessing game.
# 1. Generates and displays a random sequence of six integers.
# 2. Lets the user guess the next integer.
# 3. After submission, reveals the correct answer and indicates correctness.
# 4. Provides a button to start a new sequence.
# Simple, self‑contained, lightly styled.

SEQ_LEN = 8
MIN_VAL = 1
MAX_VAL = 20

class SequenceGame:
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

    def generate_sequence(self):
        oeis_file = '/Users/jarod/gitwork/playground/data/oeis-dataset'
        with open(oeis_file, 'r') as f:
            lines = f.readlines()
        print("Total lines in OEIS dataset:", len(lines))
        random_line = random.choice(lines).strip()
        print("Random line selected:", random_line)
        # Expected format: "A000001 ,0,1,1,1,2,1,2,1,5,2,2,1,5,1,2"

        # Split on first space only (some lines may have multiple spaces)
        if ' ' in random_line:
            a_number_part, terms_part = random_line.split(' ', 1)
        else:
            a_number_part, terms_part = random_line, ''

        a_number = a_number_part.strip()
        # Remove leading comma(s) and spaces
        terms_part = terms_part.lstrip(' ,')

        raw_terms = [t.strip() for t in terms_part.split(',') if t.strip() != '']

        # Convert to integers where possible
        terms_int = []
        for t in raw_terms:
            try:
                terms_int.append(int(t))
            except ValueError:
                # Skip non-integer tokens silently; alternatively log
                continue

        first_terms = terms_int[:SEQ_LEN]
        next_term = terms_int[SEQ_LEN] if len(terms_int) > SEQ_LEN else None
        
        print(f"First terms:", first_terms)
        print(f"Next term:", next_term)
        print("A-number:", a_number)
        return first_terms, next_term, a_number

    def new_sequence(self):
        self.sequence, self.next_term, self.a_number = self.generate_sequence()
        print("seq=",self.sequence)

        #Get Gemini's guess
        prompt = f"Given the sequence {self.sequence}, what is the next number?  Give only the number."

        client = genai.Client()
        self.gemini = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        print("gemini=",self.gemini.text)
        print("correct answer=",self.next_term)
        self.seq_label.config(text=", ".join(map(str, self.sequence)) + ", ?")
        self.result_label.config(text="Guess the next number…", fg="#203040")
        self.guess_var.set("")
        self.entry.config(state='normal')
        self.submit_btn.config(state='normal')
        self.entry.focus_set()

    def check_guess(self):
        if self.submit_btn.instate(['disabled']):
            return
        guess_text = self.guess_var.get().strip()
        try:
            guess_val = int(guess_text)
        except ValueError:
            self.result_label.config(text="Enter a valid integer.", fg="#b91c1c")
            return
        # if guess_val == self.next_term:
        #     self.result_label.config(text=f"Correct! Next number was {self.next_term}.", fg="#047857")
        # else:
        #     self.result_label.config(text=f"Incorrect. Next number was {self.next_term}.", fg="#b91c1c")
        response = f"Sequence: {self.a_number}\n"
        response += f"Your guess: {guess_val}\n"
        response += f"Gemini 2.0 Flash's guess: {self.gemini.text}\n"
        response += f"Correct answer: {self.next_term}"
        self.result_label.config(text=response, fg="#b91c1c")
        #Incorrect. Next number was {self.next_term}.", fg="#b91c1c")

        self.entry.config(state='disabled')
        self.submit_btn.state(['disabled'])


def main():
    #initialize Gemini API
    root = tk.Tk()
    SequenceGame(root)
    root.mainloop()

if __name__ == '__main__':
    main()
