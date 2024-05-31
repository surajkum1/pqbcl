import numpy as np
# from cifar100_mapping_fine_to_coarse_label import fine_to_coarse

class DataProcessing:

  def __init__(self,dataset='cifar100'):
    self.dataset = dataset
    self.selected_obj = []

  def rand_data_extractor(self):
    if self.dataset == 'CORe50':
      selected_obj_num = []

      for i in self.selected_obj:
        x,y = int(i[:2]),int(i[3:])
        if x < 3:
          z = (x-1)*50 + y

        elif x>2 and x<7:
          z = (x-2)*50 + y
        elif x!=11:
          z = (x-3)*50 + y
        else:
          z = (x-4)*50 + y
        selected_obj_num.append(z)

      rand_selected = []
      while len(rand_selected) !=5:
        x = np.random.randint(1,401,size = 1)
        if x not in selected_obj_num:
          rand_selected.append(x)


      for i in rand_selected[:5]:
        a,b = divmod(int(i),50)

        if b == 0:
          a,b = a-1,50

        if a < 2:
          s = '0'+str(a+1)
        elif a>1 and a< 5:
          s = '0'+str(a+2)
        elif a!=7:
          s = '0'+str(a+3)
        else:
          s = 11

        b = '0'+str(b) if b <10 else str(b)


        rand_selected.append(str(s)+'_'+ b)

      return rand_selected[5:]
    
    elif self.dataset =='cifar100':
      selected_obj_num = []

      for i in self.selected_obj:
        selected_obj_num.append(i)

      rand_selected = []
      while len(rand_selected) !=5:
        x = np.random.randint(0,100,size = 1)
        if x not in selected_obj_num:
          rand_selected.append(x)

      return rand_selected
    
    elif self.dataset =='cifar100_2':
      selected_obj_num = []

      for i in self.selected_obj:
        x,y = int(i[1:3]),int(i[-1:])
        z = 2*x + y
        selected_obj_num.append(z)

      rand_selected = []
      while len(rand_selected) !=5:
        x = np.random.randint(1,201,size = 1)
        if x not in selected_obj_num:
          rand_selected.append(x)


      for i in rand_selected[:5]:
        a,b = divmod(int(i),2)

        if b == 0:
          a,b = a-1,2

        if a < 10:
          a = '0' + str(a)
        else:
          a = str(a)

        rand_selected.append('o'+a+'p'+ str(b))
      return rand_selected[5:]

  
  def data_selector(self,features,paths,season=1,object=1,part = 1):
    data = []
    if self.dataset == 'CORe50':
      for i,path in enumerate(paths):
        s_n = int(path[-12:-14:-1][::-1]) # Session Number
        o_n = int(path[-9:-11:-1][::-1])  # Object Number

        if season == s_n and object == o_n:
          data.append(features[i])
      class_no = (object-1)//5
      return np.array(data),class_no*np.ones(len(data))
    
    elif self.dataset == 'cifar100':
      
      for i,o_n in enumerate(paths):
        if o_n == object:
          data.append(features[i])
      
      class_no = fine_to_coarse()[int(object)]
      return np.array(data),class_no*np.ones(len(data))
      
    elif self.dataset == 'cifar100_2':
      
      for i,o_n in enumerate(paths):
        if o_n == object:
          data.append(features[i])

          
      
      class_no = fine_to_coarse()[object]
          
        
      if part ==1:
        return np.array(data[0:250]),class_no*np.ones(len(data[0:250]))
      elif part == 2:
        return np.array(data[250:]),class_no*np.ones(len(data[250:]))