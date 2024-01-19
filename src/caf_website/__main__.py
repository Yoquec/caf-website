from .__init__ import demo

def main():
    demo.launch()

def publish():
    demo.launch(share=True)

if __name__ == '__main__':
    main() 
