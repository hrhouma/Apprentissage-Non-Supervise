### Résumé sur Keras Applications
# RÉFÉRENCE :
- https://keras.io/api/applications/
**À propos de Keras Applications**
Keras Applications fournit des modèles de deep learning pré-entraînés disponibles pour la prédiction, l'extraction de caractéristiques et le fine-tuning. Les poids des modèles sont automatiquement téléchargés lors de l'instanciation et stockés dans le répertoire `~/.keras/models/`.

**Modèles Disponibles**
- **Xception**
- **VGG16 et VGG19**
- **ResNet et ResNetV2**
- **MobileNet, MobileNetV2 et MobileNetV3**
- **DenseNet**
- **NasNetLarge et NasNetMobile**
- **InceptionV3 et InceptionResNetV2**
- **EfficientNet B0 à B7 et EfficientNetV2 B0 à B3 et S, M, L**
- **ConvNeXt Tiny, Small, Base, Large, XLarge**

**Exemples d'Utilisation**
1. **Classification des Classes ImageNet avec ResNet50**:
    ```python
    from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
    import numpy as np
    model = ResNet50(weights='imagenet')
    img_path = 'elephant.jpg'
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    ```

2. **Extraction de Caractéristiques avec VGG16**:
    ```python
    from keras.applications.vgg16 import VGG16, preprocess_input
    import numpy as np
    model = VGG16(weights='imagenet', include_top=False)
    img_path = 'elephant.jpg'
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    ```

3. **Extraction de Caractéristiques d'une Couche Intermédiaire avec VGG19**:
    ```python
    from keras.applications.vgg19 import VGG19, preprocess_input
    from keras.models import Model
    import numpy as np
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    img_path = 'elephant.jpg'
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block4_pool_features = model.predict(x)
    ```

4. **Fine-tuning d'InceptionV3 sur un Nouvel Ensemble de Classes**:
    ```python
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(200, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(...)
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    model.fit(...)
    ```

**Conclusion**
Keras Applications facilite l'utilisation de modèles pré-entraînés pour diverses tâches de deep learning, offrant des exemples pratiques et des capacités d'adaptation pour de nouvelles applications.
