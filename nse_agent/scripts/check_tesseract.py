import shutil
import sys

print('python_executable=', sys.executable)

try:
    import pytesseract
    print('pytesseract_import=OK')
    print('tesseract_cmd_attr=', getattr(pytesseract.pytesseract, 'tesseract_cmd', None))
    try:
        print('pytesseract_version=', pytesseract.get_tesseract_version())
    except Exception as e:
        print('pytesseract_get_version_error=', e)
except Exception as e:
    print('pytesseract_import_error=', e)

print('which_tesseract=', shutil.which('tesseract'))
