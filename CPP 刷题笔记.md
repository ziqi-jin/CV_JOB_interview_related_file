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
  // ret 是 二维 vector，在赋值里面的vector之前先放进去一个空的
  ret.push_back(vector <int> ());
  //清空vector
  v.clear() //一般用于给二维数组赋值之后
  // vector 的最大值最小值函数
  max_element(v.begin(),v.end())
  
  min_element(v.begin(),v.end())
  
  //1）vector容器
  
  ///例 vector<int> v;
  
  最大值：int maxValue = *max_element(v.begin(),v.end()); 
  
  最小值：int minValue = *min_element(v.begin(),v.end());
  
  //2）普通数组
  
  //例 a[]={1,2,3,4,5,6};
  
  最大值：int maxValue = *max_element(a,a+6); 
  
  最小值：int minValue = *min_element(a,a+6);
  
  2.求数组最大值最小值对应的下标
  
  1）vector容器
  
  例 vector<int> v;
  
  最大值下标：int maxPosition = max_element(v.begin(),v.end()) - v.begin(); 
  
  最小值下标：int minPosition = min_element(v.begin(),v.end()) - v.begin();
  
  2）普通数组
  
  例 a[]={1,2,3,4,5,6};
  
  最大值下标：int maxPosition = max_element(a,a+6) - a; 
  
  最小值下标：int minPosition = min_element(a,a+6) - a;
  ```

## String 

```c++
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

  String 可以用 + 连接
  // 接受 string 类型
  string str;
  getline(cin,str)
//cin.get() 可以用来处理换行符，接受不需要的字符
  
  string str1 = "dafd";
	string a = str1+str1+",";
str1.length()//返回长度
  //字符串比较直接用 > < 等
  
```

## 队列

```c++
#include<queue>
// queue<type> name;
// queue<type*> name ; 注意接受的是指针还是其他的。
queue<int> q;
// 判空
q.empty();
//队首 队尾 元素 type
q.front();
q.bacl();
// 入队，出队
q.push(i)
q.pop()// 没有返回值

```

## 排序

```c++
#include<algorithm>
// 返回bool值
bool cmp(int x,int y){
	return x % 10 > y % 10;
}

int num[10] = {65,59,96,13,21,80,72,33,44,99};
sort(num,num+10,cmp);


class Solution {
    public:
    static bool cmp(const vector<int>&a,const vector<int>&b){
        return a[1]>b[1]||(a[1]==b[1]&&a[0]>b[0]);
    }
    vector<int> filterRestaurants(vector<vector<int>>& restaurants, int veganFriendly, int maxPrice, int maxDistance) {
        int size=restaurants.size();
        vector<vector<int>>res;
        for(int i=0;i<size;i++)
        {
            if((veganFriendly==1&&restaurants[i][2]==0)||restaurants[i][3]>maxPrice||restaurants[i][4]>maxDistance)
            {
                continue;
            }
            res.push_back(restaurants[i]);
        }
        sort(res.begin(),res.end(),cmp);
        size=res.size();
        vector<int>ans;
        for(int i=0;i<size;i++){
                ans.push_back(res[i][0]);
        }
        return ans;
    }
};

```

