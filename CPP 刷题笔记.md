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
    iter->fisrt
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

## String 

```python
#include<string>
#include<string.h>
string 名字("字符串");
cout<<名字<<endl;
字符串的构造函数创建一个新字符串，包括: 
以length为长度的ch的拷贝（即length个ch）
以str为初值 (长度任意), 
以index为索引开始的子串，长度为length, 或者 
以从start到end的元素为初值. 
  string();
  string( size_type length, char ch );
  string( const char *str );
  string( const char *str, size_type length );
  string( string &str, size_type index, size_type length );
  string( input_iterator start, input_iterator end );
 

    string str1(5,'c');	//以length为长度的ch的拷贝（即length个ch）
    cout << str1 << endl;   // ccccc

    string str2( "abcde" );	//以str为初值 (长度任意),
    cout << str2 << endl;   //abcde

	string str2 = ("ILoveYou");
    string str3( str2, 1, 4 );	//str2串的下标1~4 -> "Love"
    cout << str3 << endl;	//Love

	string str4 = "123456";
	cout << str4 << endl;	//123456

	string str5;
	str5 = "654321";
	cout << str5 << endl;	//654321

	string str6 = str5;
	string str7(str5);
	cout << str6 << endl;	//654321
	cout << str7 << endl;	//654321

```

