
(cl:in-package :asdf)

(defsystem "base-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "RosImage" :depends-on ("_package_RosImage"))
    (:file "_package_RosImage" :depends-on ("_package"))
  ))