from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
		# in python3, super(Class, self).xxx = super().xxx
        super(MainWindow_controller, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.setText('open file')
        self.ui.pushButton.clicked.connect(self.pushButton_handler)
    def pushButton_handler(self):
        self.ui.textEdit_2.setText('click')
    def open_file(self):
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        print(filename, filetype)
        self.ui.show_file_path.setText(filename)

    def open_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")                 # start path
        print(folder_path)
        self.ui.show_folder_path.setText(folder_path)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())