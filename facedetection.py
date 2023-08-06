# Importing pyplot to plot images used
from matplotlib import pyplot# Importing rectangle from matplotlib to trace a rectangle on the image
from matplotlib.patches import Rectangle# Importing circle from matplotlib to trace 5 landmarks (Eyes, Nose and endpoints of the mouth)
from matplotlib.patches import Circle# Importing the main MTCNN package
from mtcnn.mtcnn import MTCNN

file_name = 'C:\Users\MAHE\Desktop\Facedetection\istockphoto-1146473249-612x612.jpg'# load image from file and plot on the graph 
plot_image = pyplot.imread(file_name)

detector = MTCNN()# This internal method detects faces in the image
face_detection = detector.detect_faces(plot_image)

print(face_detection)

# Function to trace rectangle and landmark points
def trace_face(plot_image, result_list):
  # Plots the image
    pyplot.imshow(plot_image)
    
    # Create the current axes for drawing boxes
    axes = pyplot.gca()
    for result in result_list:
        ''' Get the coordinates of the rectangle from result for
            every face detected by MTCNN() '''
        x, y, width, height = result['box']
        ''' Call the rectangular function using above coordinate
            values '''
        traced_rectangle = Rectangle((x, y), width, height,
        fill=False, color='red')
        # Draw the rectangle encasing the faces
        axes.add_patch(traced_rectangle)

        for key, value in result['keypoints'].items():
            # Call the circle function for coordinate values
            key_points = Circle(value, radius=2, color='red')
            # Draw the circular points on the faces
            axes.add_patch(key_points)

# Function to trace rectangle and landmark points
trace_face(plot_image, face_detection)

# Plot the faces
pyplot.show()
    