# CPP 刷题笔记

## vector 长度 

- 输入为一个 vector 数组nums, 要循环就

  ```c++
  int n = nums.size();
  for (int i=0;i<n;i++){
    nums[i]....
  }
  ```

## 哈希表

- ```c++
  //头文件
  #include<unordered_map>
  //定义
  unordered_map<int,int> hashtable;
  unordered_map<key_type,val_type> name = {{,},{,}}
  // 赋值，插入
  name["key"] = val
  //判断是否包含key,这两个都是没有的标志
  if (name.count(key)==0)
  if (name.find(key)==name.end())
  // find会返回一个 迭代器，迭代器有2个值，fisrt：key，second：val
  iter = name.find()
    iter->fisdt
    iter->second
    
  ```

## Vector 数组

- ```c++
  #include<vector>
  
  vector<int> v = {0};
  v.push_back(val)//插入一个值，在尾部
  v.pop_back()//弹出尾部值
  v.size()//获取长度
    
  ```

- 

