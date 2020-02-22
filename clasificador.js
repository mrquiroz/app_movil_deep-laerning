import React, { Component } from 'react';
import ImagePickerManager from 'react-native-image-picker';
import ImagePicker from 'react-native-image-crop-picker';
import { Platform, StyleSheet, Image,Modal, Text, View,TouchableOpacity, Alert,ActivityIndicator} from 'react-native';
import ScalableText from 'react-native-text';
import AwesomeAlert from 'react-native-awesome-alerts';
import { createKeyboardAwareNavigator } from 'react-navigation';

const height = 350;
const width = 350;
const blue = "#25d5fd";


export default class Clasificador extends Component {

    constructor(props) {
        super(props);
        this.state = {
            model: true,
            source: null,
            loading: false,
            imageHeight: height,
            imageWidth: width,
            showAlert:false,
            modalVisible:false,
            recognitions: [],
            mensaje : '',
            color_mensaje: '',
            clasificacion:'',
          };
    }

      clasificar() {
        const options = {
          takePhotoButtonTitle: "Tomar una fotografia",
          cancelButtonTitle: "Cancelar",
          title: "Selecciona una opcion",
          chooseFromLibraryButtonTitle:"Foto desde libreria",
          storageOptions: {
            skipBackup: true,
            path: 'images',
          },
        };

      
      ImagePickerManager.showImagePicker(options, (response) => {
          if (response.didCancel) {
            console.log('User cancelled image picker');
          } else if (response.error) {
            console.log('ImagePicker Error: ', response.error);
          } else if (response.customButton) {
            console.log('User tapped custom button: ', response.customButton);
          } else {
            ImagePicker.clean().then(() => {
              console.log('removed all tmp images from tmp directory');
            }).catch(e => {
              alert(e);
            });
            ImagePicker.openCropper({
              path: 'file://' + response.path,
              width: 224,
              height: 224,
              includeBase64:true
            }).then(image => {
              this.setState({loading:true})
              fetch('http://54.237.193.93/predict_m', {
                      method: 'POST',
                      body: JSON.stringify({
                        data:image.data
                      }),
                  })
                  .then(response => response.json())
                  .then(response => {
                    this.setState({clasificacion:response["class_name"],modalVisible:true})
                    console.log(response["class_name"])
                  if (response["class_name"]== 'Melanoma') {
                      this.setState({color_mensaje:'red'})
                      this.setState({mensaje:'Dados nuestro analisis, te recomendamos ir donde un especialista'})
                  } 
                  else{
                    this.setState({color_mensaje:'green'})
                    this.setState({mensaje:'No hemos encontrado nada raro en tu lunar'})
                  }
                    this.setState({loading:false})
                    this.setState({showAlert:true})
                    this.setState({
                      source: { uri: image.path },
                      imageHeight: 224,
                      imageWidth: 224
                    });
                  })
            });
          }
        });
      }
      
    renderResults() {
        const { model, recognitions, imageHeight, imageWidth } = this.state;
        
            return recognitions.map((res, id) => {
              return (
                <Modal transparent={true}
                  visible={this.state.modalVisible}
                  onRequestClose={
                    ()=>{this.props.navigation.navigate('Login')
                }}>
          
                <Text key={id} style={{ color: 'black',backgroundColor: 'white',alignContent:"center",paddingTop:30,textAlign:"center" }}>
                        {res["label"] + "-" + (res["confidence"] * 100).toFixed(0) + "%"}
                      </Text>
                      <Text style={{position: 'absolute',
      left:     10,
      bottom: 30,}}>Lo anterior se trata de una aproximacion y en caso de duda visita a un especialista</Text>
              </Modal> 
                    )
                  });
        
            }
      render() {
        const { model, source, imageHeight, imageWidth } = this.state;
        var renderButton = (m) => {
          
          return (
            <TouchableOpacity style={styles.button} onPress={this.clasificar.bind(this)}>
              <Text style={styles.buttonText}>{m}</Text>
            </TouchableOpacity>
          );
        }
        return (
          <View style={styles.container}>
          <ScalableText style={styles.text_2}>Recuerda que el resultado se trata de una aproximacion, en caso de dudas tocando la imagen puedes intenter nuevamente</ScalableText>
                     
<Loader
          loading={this.state.loading} />
<Modal 
                transparent={true}
                visible={this.state.showAlert}
            >
            <AwesomeAlert
                show={this.state.showAlert}
                showProgress={false}
                title="Resultado clasificacion"
                message= {this.state.mensaje}
                closeOnTouchOutside={false}
                closeOnHardwareBackPress={false}
                showConfirmButton={true}
                confirmText="Entendido"
                confirmButtonColor={this.state.color_mensaje}
                onConfirmPressed={() => this.setState({showAlert:false})}
              />
              </Modal>
            {model ?
              <TouchableOpacity style={
                [styles.imageContainer, {
                  height: imageHeight,
                  width: imageWidth,
                  borderWidth: source ? 0 : 2
                }]} onPress={this.clasificar.bind(this)}>
                {
                  source ?
                    <Image source={source} style={{
                      height: imageHeight, width: imageWidth
                    }} resizeMode="contain" /> :
                    <Text style={styles.text}>Selecciona alguna imagen</Text>
                }
                <View style={styles.boxes}>
                  {this.renderResults()}
                </View>
              </TouchableOpacity>
              :
              <View>
                {renderButton('Imagen')}
              </View>
            }
          </View>
         
        );
      }
    }

    const Loader = props => {
      const {
        loading,
        ...attributes
      } = props;
    
      return (
        <Modal
          transparent={true}
          animationType={'none'}
          visible={loading}
          onRequestClose={() => {console.log('close modal')}}>
          <View style={styles.modalBackground}>
            <View style={styles.activityIndicatorWrapper}>
              <ActivityIndicator
                animating={loading} />
            </View>
          </View>
        </Modal>
      )
    }

    let Loader_2 = props => {
      var {
        loading,
        mensaje,
        ...attributes
      } = props;
   
      return (
        <Modal 
                transparent={true}
                visible={loading}
            >
            <AwesomeAlert
                show={loading}
                showProgress={false}
                title="Resultado clasificacion"
                message= "El resultado"
                closeOnTouchOutside={true}
                closeOnHardwareBackPress={true}
                showConfirmButton={true}
                confirmText="ok"
                confirmButtonColor="red"
                onConfirmPressed={() => loading = false}
              />
              </Modal>
      )
    }
    
    const styles = StyleSheet.create({
      container: {
        flex: 1,
        alignItems: 'center',
        backgroundColor: 'white'
      },
      imageContainer: {
        borderColor: blue,
        borderRadius: 5,
        alignItems: "center"
      },
      modalBackground: {
        flex: 1,
        alignItems: 'center',
        flexDirection: 'column',
        justifyContent: 'space-around',
        backgroundColor: '#00000040'
      },
      activityIndicatorWrapper: {
        backgroundColor: '#FFFFFF',
        height: 100,
        width: 100,
        borderRadius: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-around'
      },
      text: {
        fontSize: 15,
        textAlign: 'center',
        color: '#333333',
        marginBottom: 5,
        marginTop: 145,
        marginLeft:20,
        marginRight:20,
      },
      text_2: {
        fontSize: 19,
        textAlign: 'center',
        color: '#333333',
        marginBottom: 30,
        marginTop: 40,
        marginLeft:20,
        marginRight:20,
      },
      myButton:{ 
        backgroundColor:'#68A7B8',
        borderRadius:400,
        borderColor: '#d6d7da',
        
      },
      button: {
        width: 200,
        backgroundColor: blue,
        borderRadius: 10,
        height: 40,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 10
      },
      background: {
        flex:1,
        width: '100%', 
        height: '101%',
        resizeMode: 'cover',
      },
      buttonText: {
        color: 'white',
        fontSize: 15
      },
      box: {
        position: 'absolute',
        borderColor: blue,
        borderWidth: 2,
      },
      boxes: {
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 0,
        top: 0,
      }
    });