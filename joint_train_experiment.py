from notebook import *

def find_and_pop_arg(arg_name: str):
    for a in sys.argv:
        if a.startswith(f"--{arg_name}"):
            sys.argv.remove(a)
            val = a.split("=",1)[1]
            return val 

if __name__ == "__main__":
    qanom_joint_factor =find_and_pop_arg("qanom_joint_factor")
    test_task =find_and_pop_arg("test_task")
    dataset_arg = get_joint_params(test_task, qanom_joint_factor)
    sys.argv.extend(dataset_arg)
    main()
        
        