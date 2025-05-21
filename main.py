try:
    from utils import create_dashboard
    print("Import success")
except ImportError as e:
    print("Import failed")
    print(e)

create_dashboard()