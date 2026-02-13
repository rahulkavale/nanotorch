from nanotorch.plotting import main


# Keep a tiny wrapper so the script path remains valid, while the real\n+# logic lives in the library and can be exposed as a console entrypoint.\n+if __name__ == "__main__":\n+    main()
