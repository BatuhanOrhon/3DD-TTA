from pathlib import Path
import os
os.chdir('/content/3DD-TTA' if Path('/content/3DD-TTA').is_dir() else '/content')
!pip install -q condacolab
import condacolab
condacolab.install()
