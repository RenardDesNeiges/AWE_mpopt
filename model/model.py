from utils.config_loader import ConfigParser

class Model:
    
    def __init__(self, config_file):
        
        def rec_yaml(y_dict):    
            for key,value in y_dict.items():
                if type(value) is not dict:
                    setattr(self, key, value)
                else:
                    rec_yaml(value)

        
        self._config = ConfigParser.load(config_file)
        rec_yaml(self._config)
        
    def dyn_forward(self,x,u):
        pass # TODO
    
    def dyn_f(self,x,u):
        print("Dynamics not implemented")
        pass 

        