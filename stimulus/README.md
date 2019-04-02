I used python3 virtual environments
I use a mac, but it might work on windows easily:


First to install python 3 use homebrew
install homebrew:

```ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"```

install python3:

```brew install python3```

pull the repo:

```git clone https://github.com/APPLabUofA/BCI_Python```

create a virtual environment and activate:
```
python3 -m venv venv
source venv/bin/activate
```

Install all the requirements:
```
pip install --upgrade pip
pip install -r requirements.txt
```

Then you should be able to run the task with:
```
python simple.py
```


