import tkinter as tk
from tkinter import *
import os
from tkinter import messagebox
from test_main_easy import start_display as start_display_easy
from test_main_medium import start_display as start_display_medium
from test_main_hard import start_display as start_display_hard

class DotAndBoxGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Dots and Boxes")
        self.root.geometry("500x500+1000+300")
        self.root.resizable(False, False)
        icon_path = os.path.join(os.path.dirname(__file__), 'images', 'dotsandboxes.ico')
        self.root.iconbitmap = icon_path

        self.selected_mode = None
        self.selected_level = None
        self.create_main_menu()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_main_menu(self):
        self.clear_window()

        container = tk.Frame(self.root)
        container.pack(expand=True, fill='both')

        container.columnconfigure(0, weight=1)

        container.rowconfigure(0, weight=2)
        container.rowconfigure(1, weight=1)
        container.rowconfigure(2, weight=1)
        container.rowconfigure(3, weight=1)
        container.rowconfigure(4, weight=2)

        title = tk.Label(container, text="Dots and Boxes", fg="red", font=("Comfortaa", 30))
        title.grid(row=0, column=0)

        mode = tk.Label(container, text="Chọn chế độ:", font=("VNI-Dom", 18))
        mode.grid(row=1, column=0)

        btn_person_vs_ai = tk.Button(container, text="Person vs AI", font=("VNI-Dom", 14),
                                     command=lambda: self.set_mode("Person vs AI"),
                                     bd="4", bg="light blue", fg="purple", width="10", pady="5")
        btn_person_vs_ai.grid(row=2, column=0)

        btn_ai_vs_ai = tk.Button(container, text="AI vs AI", font=("VNI-Dom", 14),
                                 command=lambda: self.set_mode("AI vs AI"),
                                 bd="4", bg="light green", fg="darkred", width="10", pady="5")
        btn_ai_vs_ai.grid(row=3, column=0)

    def set_mode(self, mode):
        self.selected_mode = mode
        if mode == "Person vs AI":
            self.select_level()
        else:
            self.select_board_size()

    def set_level_and_continue(self, level):
        self.selected_level = level
        self.select_board_size()

    def select_level(self):
        self.clear_window()
        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both")

        container.columnconfigure(0, weight=1)

        for i in range(5):
            container.rowconfigure(i, weight=1)

        label = tk.Label(container, text="Chọn độ khó:", font=("VNI-Dom", 16))
        label.grid(row=0, column=0)

        btn_easy = tk.Button(container, text="Dễ", font=("VNI-Dom", 14),
                             command=lambda: self.set_level_and_continue("easy"),
                             bd="6", bg="light pink", fg="blue")
        btn_easy.grid(row=1, column=0)

        btn_medium = tk.Button(container, text="Vừa", font=("VNI-Dom", 14),
                               command=lambda: self.set_level_and_continue("medium"),
                               bd="6", bg="light pink", fg="blue")
        btn_medium.grid(row=2, column=0)

        btn_hard = tk.Button(container, text="Khó", font=("VNI-Dom", 14),
                             command=lambda: self.set_level_and_continue("hard"),
                             bd="6", bg="light pink", fg="blue")
        btn_hard.grid(row=3, column=0)

        btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.create_main_menu)
        btn_back.pack(side=tk.BOTTOM, pady=10)

    def select_board_size(self):
        self.clear_window()
        
        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both")

        container.columnconfigure(0, weight=1)

        container.rowconfigure(0, weight=2)  
        container.rowconfigure(1, weight=1) 
        container.rowconfigure(2, weight=1) 
        container.rowconfigure(3, weight=1)  
        container.rowconfigure(4, weight=1)
        container.rowconfigure(5, weight=1)
        container.rowconfigure(6, weight=2) 
        
        label = tk.Label(container, text= "Chọn kích thước bảng:", font=("VNI-Dom", 16))
        label.grid(column=0,row=1)
        
        sizes = ["3x3", "4x4", "4x5","5x5"]
        for size in sizes:
            if (size == "3x3"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s,self.selected_mode), bd="5", bg="Green", fg="Yellow") 
                btn.grid(column=0,row=2)
            if (size == "4x4"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s, self.selected_mode), bd="5", bg="Yellow")
                btn.grid(column=0,row=3)
            if (size == "4x5"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s, self.selected_mode), bd="5", bg="#F4A950", fg="blue")
                btn.grid(column=0,row=4)
            if (size == "5x5"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s, self.selected_mode), bd="5", bg="#ff5b61", fg="light yellow")
                btn.grid(column=0,row=5)
                
                def blink():
                    current_color = btn.cget("fg")
                    new_color = "cyan" if current_color == "light yellow" else "light yellow"
                    btn.config(fg=new_color)
                    btn.after(500, blink)

                blink()  # Hiệu ứng nhấp nháy
            # btn.pack(pady=20)

        btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.select_level)
        btn_back.pack(side=tk.BOTTOM, pady=10)

    def initiate_game_start(self, board_size_str, mode):
        try:
            self.root.withdraw()
            if mode == "Person vs AI":
                if self.selected_level == "easy":
                    start_display_easy(board_size_str, mode)
                elif self.selected_level == "medium":
                    start_display_medium(board_size_str, mode)
                elif self.selected_level == "hard":
                    start_display_hard(board_size_str, mode)
                else:
                    raise ValueError("Chưa chọn mức độ khó.")
            else:
                start_display_easy(board_size_str, mode)
        except Exception as e:
            messagebox.showerror("Error", f"Could not start game: {e}")
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.deiconify()
                    self.create_main_menu()
            except tk.TclError:
                print("Tkinter window was closed.")

# Chạy chương trình
if __name__ == "__main__":
    root = tk.Tk()
    app = DotAndBoxGame(root)
    root.mainloop()
