class ConsoleLogHandler:
    def __init__(self, console_widget):
        self.console_widget = console_widget

    def write(self, message):
        # Filter out empty messages (can occur when flushing)
        if message.strip():
            self.console_widget.append(message)
            # Scroll to the bottom of the log
            self.console_widget.moveCursor(self.console_widget.textCursor().End)

    def flush(self):
        # This is required for the interface, but does nothing
        pass
