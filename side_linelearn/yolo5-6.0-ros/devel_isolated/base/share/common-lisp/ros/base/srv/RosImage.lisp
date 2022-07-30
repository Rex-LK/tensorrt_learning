; Auto-generated. Do not edit!


(cl:in-package base-srv)


;//! \htmlinclude RosImage-request.msg.html

(cl:defclass <RosImage-request> (roslisp-msg-protocol:ros-message)
  ((image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass RosImage-request (<RosImage-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RosImage-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RosImage-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name base-srv:<RosImage-request> is deprecated: use base-srv:RosImage-request instead.")))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <RosImage-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base-srv:image-val is deprecated.  Use base-srv:image instead.")
  (image m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RosImage-request>) ostream)
  "Serializes a message object of type '<RosImage-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RosImage-request>) istream)
  "Deserializes a message object of type '<RosImage-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RosImage-request>)))
  "Returns string type for a service object of type '<RosImage-request>"
  "base/RosImageRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RosImage-request)))
  "Returns string type for a service object of type 'RosImage-request"
  "base/RosImageRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RosImage-request>)))
  "Returns md5sum for a message object of type '<RosImage-request>"
  "222963dddca319dd44aea7d612d0ab69")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RosImage-request)))
  "Returns md5sum for a message object of type 'RosImage-request"
  "222963dddca319dd44aea7d612d0ab69")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RosImage-request>)))
  "Returns full string definition for message of type '<RosImage-request>"
  (cl:format cl:nil "sensor_msgs/Image image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RosImage-request)))
  "Returns full string definition for message of type 'RosImage-request"
  (cl:format cl:nil "sensor_msgs/Image image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RosImage-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RosImage-request>))
  "Converts a ROS message object to a list"
  (cl:list 'RosImage-request
    (cl:cons ':image (image msg))
))
;//! \htmlinclude RosImage-response.msg.html

(cl:defclass <RosImage-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:string
    :initform ""))
)

(cl:defclass RosImage-response (<RosImage-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RosImage-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RosImage-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name base-srv:<RosImage-response> is deprecated: use base-srv:RosImage-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <RosImage-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader base-srv:result-val is deprecated.  Use base-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RosImage-response>) ostream)
  "Serializes a message object of type '<RosImage-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'result))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'result))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RosImage-response>) istream)
  "Deserializes a message object of type '<RosImage-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'result) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'result) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RosImage-response>)))
  "Returns string type for a service object of type '<RosImage-response>"
  "base/RosImageResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RosImage-response)))
  "Returns string type for a service object of type 'RosImage-response"
  "base/RosImageResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RosImage-response>)))
  "Returns md5sum for a message object of type '<RosImage-response>"
  "222963dddca319dd44aea7d612d0ab69")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RosImage-response)))
  "Returns md5sum for a message object of type 'RosImage-response"
  "222963dddca319dd44aea7d612d0ab69")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RosImage-response>)))
  "Returns full string definition for message of type '<RosImage-response>"
  (cl:format cl:nil "string result~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RosImage-response)))
  "Returns full string definition for message of type 'RosImage-response"
  (cl:format cl:nil "string result~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RosImage-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'result))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RosImage-response>))
  "Converts a ROS message object to a list"
  (cl:list 'RosImage-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'RosImage)))
  'RosImage-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'RosImage)))
  'RosImage-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RosImage)))
  "Returns string type for a service object of type '<RosImage>"
  "base/RosImage")