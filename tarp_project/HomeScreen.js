import * as React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import Constants from 'expo-constants';
import * as Firebase from 'firebase';
import { Card } from 'react-native-paper';
import {Container,Content,Header,Form,Input,Item,Button,Label} from 'native-base'; 
const firebaseConfig = {
  apiKey: "AIzaSyAf0zJlQT3EHqpUnUKLlc_JwS_BR6zlWSg",
  authDomain: "tarp-alzheimer.firebaseapp.com",
  databaseURL: "https://tarp-alzheimer.firebaseio.com",
  projectId: "tarp-alzheimer",
  storageBucket: "tarp-alzheimer.appspot.com",
  messagingSenderId: "615755110137",
  appId: "1:615755110137:web:bb04e83d2597a1c5b3f428",
  measurementId: "G-KD1NXG9FE1"
};
var s;
var database;
if(Firebase.app.length == 0)
{
Firebase.initializeApp(firebaseConfig);
}

database = Firebase.database();

var ref = database.ref('TestData');
ref.on('value',gotData,errData)



function gotData(data)
{
  var scores = data.val();
  var keys = Object.keys(scores);
  console.log(keys);
  s = '';
  for(var i = 0;i<keys.length;i++)
  {
    var k = keys[i];
    var data1 = scores[k].data1;
    var data2 = scores[k].data2;
    if (data1 == 3)
    {
       s = s + data2;
    }
  }
}

function errData(err)
{
  console.log('Error');
  console.log(err) ;
}


export default class HomeScreen extends React.Component {
  constructor()
  {
  super();
  this.state = { 
    Object1 : '',
   }
  }
  render() {
    const { navigate } = this.props.navigation;  
    return (

     <Container style = {styles.container}>
     <Form>
     <Item floatingLabel>
     <Label>Object_for_Search</Label>
     <Input
       autoCorrect = {false}
       autoCapitalize = "none"
       value={this.state.Object1}
       onChangeText={Object1 => this.setState({ Object1 })} 
     />  
     </Item>
     <Button style = {{marginTop : 20}}
      full
      rounded
      success
      onPress={() =>  
        this.props.navigation.navigate('Search', {  
            Object1: this.state.Object1, 
        })  
    }  
     >
     <Text> Submit </Text>
     </Button>
     </Form>
     </Container>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    paddingTop: Constants.statusBarHeight,
    backgroundColor: '#0AE',
    padding: 8,
  },
  paragraph: {
    margin: 24,
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
